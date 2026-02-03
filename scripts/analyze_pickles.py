import pickletools
import sys
import os
import pandas as pd
import io

# Store summary data for the final comparison table
SUMMARY_DATA = []

def analyze_structure_bytecode(filepath):
    """
    Analyzes a pickle file using pickletools to extract GLOBAL imports
    and string literals that appear to be data type signatures.
    Returns a dict of found signatures.
    """
    stats = {
        'globals': set(),
        'dtypes': set(),
        'protocol': 'Unknown'
    }
    
    INTERESTING_STRINGS = {
        'numpy', 'pandas', 
        'int8', 'int16', 'int32', 'int64', 
        'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64', 
        'complex64', 'complex128',
        'bool', 'object', 'O8', '<i8', '<f8', '|O8',
        'StringDtype', 'StringArray', 'ArrowStringArray',
        'category', 'datetime64', 'timedelta64',
        'ArrowStringArray', 'string' # general checks
    }

    try:
        with open(filepath, 'rb') as f:
            # Check protocol version from header if possible, but genops does this
            last_strings = [None, None]
            try:
                for opcode, arg, pos in pickletools.genops(f):
                    if opcode.name == 'PROTO':
                        stats['protocol'] = arg
                    elif opcode.name == 'GLOBAL':
                        stats['globals'].add(arg)
                    elif opcode.name == 'STACK_GLOBAL':
                        if last_strings[0] and last_strings[1]:
                            guessed_global = f"{last_strings[0]} {last_strings[1]}"
                            stats['globals'].add(guessed_global)
                        else:
                            stats['globals'].add("<STACK_GLOBAL>") 
                    elif opcode.name in ('SHORT_BINSTRING', 'BINSTRING', 'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'):
                        if isinstance(arg, str):
                            last_strings.pop(0)
                            last_strings.append(arg)
                            if arg in INTERESTING_STRINGS:
                                stats['dtypes'].add(arg)
            except Exception as e:
                return {'error': str(e)}
    except Exception as e:
        return {'error': f"Could not open file: {str(e)}"}

    return stats

def inspect_data(filepath):
    """
    Attempts to load the pickle file with pandas and report high-level info.
    """
    print(f"\n{'='*60}")
    print(f"REPORT: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    filename = os.path.basename(filepath)
    file_summary = {
        'filename': filename,
        'status': 'Error',
        'uttid_dtype': '?',
        'pred_dtype': '?'
    }

    # 1. Bytecode Analysis (Always run this for metadata, even if load works)
    structure = analyze_structure_bytecode(filepath)
    if 'error' in structure:
        print(f"(!) File corrupted or unreadable: {structure['error']}")
        SUMMARY_DATA.append(file_summary)
        return

    # 2. Try Loading
    try:
        # strict=False allows loading unpickles that might rely on other modules
        # but in a script we generally want to see if it works.
        df = pd.read_pickle(filepath)
        
        print(f"Status:   SUCCESS - Loaded with pandas {pd.__version__}")
        print(f"Type:     {type(df).__name__}")
        
        file_summary['status'] = 'Loaded'

        def report_dataframe(dframe, indent=""):
            print(f"{indent}Shape:    {dframe.shape[0]} rows x {dframe.shape[1]} columns")
            
            # Formatted Column List
            print(f"\n{indent}--- Column Datatypes ---")
            max_col_len = max([len(str(c)) for c in dframe.columns]) if len(dframe.columns) > 0 else 10
            header = f"{indent}{'Column':<{max_col_len+2}} {'Dtype':<15} {'Sample Value'}"
            print(header)
            print(f"{indent}" + "-" * (len(header) - len(indent)))
            
            for col in dframe.columns:
                dtype = str(dframe[col].dtype)
                
                # Update Summary
                if 'uttid' in str(col).lower():
                    file_summary['uttid_dtype'] = dtype
                elif 'pred' in str(col).lower():
                    file_summary['pred_dtype'] = dtype

                # Get a non-null sample if possible
                sample = dframe[col].dropna().iloc[0] if not dframe[col].dropna().empty else "<All NaN>"
                # Truncate long samples
                sample_str = str(sample)
                if len(sample_str) > 50:
                    sample_str = sample_str[:47] + "..."
                print(f"{indent}{str(col):<{max_col_len+2}} {dtype:<15} {sample_str}")

            print(f"\n{indent}--- Data Sample (Head) ---")
            df_str = dframe.head(3).to_string()
            print('\n'.join([f"{indent}{line}" for line in df_str.split('\n')]))

        if isinstance(df, pd.DataFrame):
            report_dataframe(df)

        elif isinstance(df, pd.Series):
            print(f"Length:   {len(df)}")
            print(f"Dtype:    {df.dtype}")
            print("\n--- Data Sample ---")
            print(df.head(5).to_string())
        
        elif isinstance(df, dict):
            print("Content (Dictionary Keys):")
            for k, v in df.items():
                print(f"\n> Key: '{k}'")
                if isinstance(v, pd.DataFrame):
                    print(f"  Type: DataFrame")
                    report_dataframe(v, indent="  ")
                else:
                    val_str = str(v)
                    if len(val_str) > 100:
                        val_str = val_str[:97] + "..."
                    print(f"  Type: {type(v).__name__}")
                    print(f"  Value: {val_str}")
            
        else:
            print(f"Content:  {df}")

    except Exception as e:
        print(f"Status:   FAILED to load")
        print(f"Error:    {str(e)}")
        print("\n(!) This file could not be unpickled in the current environment.")
        
        file_summary['status'] = 'Failed'
        
        # INFERENCE LOGIC
        # 1. Check Error Message for Clues
        err_str = str(e)
        if 'StringDtype' in err_str:
             file_summary['uttid_dtype'] = 'StringDtype (In Error)'
        
        # 2. Check Bytecode if still unknown
        if file_summary['uttid_dtype'] == '?':
            if 'StringDtype' in structure['dtypes'] or 'StringArray' in structure['dtypes']:
                file_summary['uttid_dtype'] = 'StringDtype (Bytecode)'
            elif 'O8' in structure['dtypes']:
                file_summary['uttid_dtype'] = 'object (Bytecode)'
        
        # Assume predictions are float/numeric if we see numpy
        if file_summary['pred_dtype'] == '?':
             if 'numpy' in structure['dtypes'] or 'O8' in structure['dtypes']:
                 file_summary['pred_dtype'] = 'numeric/float (Inferred)'


    # 3. Low-level details (Technical)
    print("\n--- Internal Structure (Bytecode Analysis) ---")
    print(f"Pickle Protocol: {structure['protocol']}")
    
    if structure['dtypes']:
        print(f"Detected Types:  {sorted(list(structure['dtypes']))}")
    else:
        print(f"Detected Types:  (None explicitly named in string literals)")
        
    print(f"Key Imports:     ")
    if structure['globals']:
        for g in sorted(list(structure['globals'])):
            print(f"  - {g}")
    else:
        print("  (No explicit imports found)")
    print("\n")
    
    SUMMARY_DATA.append(file_summary)


def print_summary_table():
    print(f"\n{'='*80}")
    print(f"{'DATATYPE COMPARISON SUMMARY':^80}")
    print(f"{'='*80}")
    
    # Headers
    headers = ["Filename", "Status", "Col: uttid", "Col: predictions"]
    row_fmt = "{:<40} {:<10} {:<20} {:<20}"
    
    print(row_fmt.format(*headers))
    print("-" * 80)
    
    for row in SUMMARY_DATA:
        # Shorten filename
        fname = row['filename']
        if len(fname) > 38:
            fname = fname[:35] + "..."
            
        print(row_fmt.format(
            fname,
            row['status'],
            str(row['uttid_dtype']),
            str(row['pred_dtype'])
        ))
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pickles.py <file1> <file2> ...")
        sys.exit(1)
    
    for f in sys.argv[1:]:
        if os.path.exists(f):
            inspect_data(f)
        else:
            print(f"File not found: {f}")
            
    print_summary_table()