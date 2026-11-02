#!/usr/bin/env python3
"""TOPSIS engine for multi-criteria cloud provider evaluation."""

import csv
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PROVIDERS = ["AWS", "Azure", "GCP"]
CRITERIA = ["performance", "cost", "reliability", "security", "scalability"]
CRITERIA_TYPES = ["benefit", "cost", "benefit", "benefit", "benefit"]

# Decision matrix — Table 2 in the thesis.
# Rows: AWS, Azure, GCP
# Cols: performance (MLPerf score 0-100), cost ($/GPU/hr H100 on-demand),
#       reliability (SLA uptime %), security (cert score 0-100),
#       scalability (infra breadth 0-100)
DECISION_MATRIX = np.array([
    [85,  3.90, 99.95, 100, 95],  # AWS
    [80,  6.98, 99.90,  95, 92],  # Azure
    [90,  3.00, 99.95,  95, 90],  # GCP
])

BASELINE_WEIGHTS = np.array([0.25, 0.30, 0.20, 0.15, 0.10])


def topsis(matrix, weights, criteria_types):
    """
    Run TOPSIS (Hwang & Yoon, 1981).

    Returns closeness coefficients, D+ and D- for each alternative.
    """
    w = weights / weights.sum()
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1
    weighted = (matrix / norms) * w

    ideal = np.where(
        np.array([t == "benefit" for t in criteria_types]),
        weighted.max(axis=0),
        weighted.min(axis=0),
    )
    anti_ideal = np.where(
        np.array([t == "benefit" for t in criteria_types]),
        weighted.min(axis=0),
        weighted.max(axis=0),
    )

    d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    denom = d_plus + d_minus
    denom[denom == 0] = 1
    closeness = d_minus / denom

    return closeness, d_plus, d_minus


def rank_providers(weights=None):
    """Run TOPSIS on thesis data and return ranked results."""
    if weights is None:
        weights = BASELINE_WEIGHTS.copy()
    scores, dp, dm = topsis(DECISION_MATRIX, weights, CRITERIA_TYPES)
    order = np.argsort(-scores)
    return [
        {
            "rank": int(pos + 1),
            "provider": PROVIDERS[i],
            "score": float(scores[i]),
            "d_plus": float(dp[i]),
            "d_minus": float(dm[i]),
        }
        for pos, i in enumerate(order)
    ]


def load_matrix_from_csv(path=None):
    """Load decision matrix from CSV file."""
    if path is None:
        path = DATA_DIR / "mcda_dataset.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    providers = [r["provider"] for r in rows]
    matrix = np.array([
        [float(r[c]) for c in CRITERIA] for r in rows
    ])
    return providers, matrix


if __name__ == "__main__":
    results = rank_providers()
    print(f"{'Rank':<6}{'Provider':<10}{'C_i':<12}{'D+':<12}{'D-':<12}")
    print("-" * 52)
    for r in results:
        print(f"{r['rank']:<6}{r['provider']:<10}{r['score']:<12.4f}"
              f"{r['d_plus']:<12.4f}{r['d_minus']:<12.4f}")
