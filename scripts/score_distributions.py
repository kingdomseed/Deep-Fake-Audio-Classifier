import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Summary:
    rows: int
    min: float
    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float
    frac_lt_001: float
    frac_gt_099: float
    frac_mid_01_09: float


def summarize(path: str) -> Summary:
    df = pd.read_pickle(path)
    scores = df["predictions"].astype(float).to_numpy()

    q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.quantile(
        scores, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).tolist()

    return Summary(
        rows=int(scores.shape[0]),
        min=float(scores.min()),
        p01=float(q01),
        p05=float(q05),
        p10=float(q10),
        p25=float(q25),
        p50=float(q50),
        p75=float(q75),
        p90=float(q90),
        p95=float(q95),
        p99=float(q99),
        max=float(scores.max()),
        frac_lt_001=float((scores < 0.01).mean()),
        frac_gt_099=float((scores > 0.99).mean()),
        frac_mid_01_09=float(((scores >= 0.1) & (scores <= 0.9)).mean()),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score distribution summary for prediction.pkl")
    p.add_argument("paths", nargs="+", help="One or more prediction .pkl paths")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "name,rows,min,p01,p05,p10,p25,p50,p75,p90,p95,p99,max,frac_lt_0.01,frac_gt_0.99,frac_mid_0.1_0.9"
    )
    for path in args.paths:
        s = summarize(path)
        d = asdict(s)
        name = path
        print(
            f"{name},{d['rows']},{d['min']:.6g},{d['p01']:.6g},{d['p05']:.6g},{d['p10']:.6g},{d['p25']:.6g},{d['p50']:.6g},{d['p75']:.6g},{d['p90']:.6g},{d['p95']:.6g},{d['p99']:.6g},{d['max']:.6g},{d['frac_lt_001']:.3f},{d['frac_gt_099']:.3f},{d['frac_mid_01_09']:.3f}"
        )


if __name__ == "__main__":
    main()
