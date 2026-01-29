import sys
import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise ValueError("Usage: python generate_submission.py <features.pkl> <prediction.pkl> <Student_ID> <FirstName> <LastName> <Nickname>")
    
    features_path = sys.argv[1]
    prediction_path = sys.argv[2]
    student_id = sys.argv[3]
    first_name = sys.argv[4]
    last_name = sys.argv[5]
    nickname = sys.argv[6]
    
    features_df = pd.read_pickle(features_path)
    prediction_df = pd.read_pickle(prediction_path)
    
    if len(prediction_df.columns) != 2:
        raise ValueError("prediction.pkl must have exactly 2 columns")
    
    if 'uttid' not in prediction_df.columns or 'predictions' not in prediction_df.columns:
        raise ValueError("prediction.pkl must have 'uttid' and 'predictions' columns")
    
    if 'uttid' not in features_df.columns:
        raise ValueError("features.pkl must have 'uttid' column")
    
    features_uttids = set(features_df['uttid'].values)
    prediction_uttids = set(prediction_df['uttid'].values)
    
    if features_uttids != prediction_uttids:
        raise ValueError("uttid mismatch between features.pkl and prediction.pkl")
    
    if not all(isinstance(x, (float, np.floating)) for x in prediction_df['predictions'].values):
        prediction_df['predictions'] = prediction_df['predictions'].astype(np.float64)
    
    result = {
        "student_id": student_id,
        "first_name": first_name,
        "last_name": last_name,
        "nickname": nickname,
        "predictions": prediction_df
    }
    
    output_filename = f"{student_id}-{first_name}-{last_name}-{nickname}.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Submission file saved to: {output_filename}")
