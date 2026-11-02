#!/usr/bin/env python3
"""
Sensitivity analysis for TOPSIS cloud provider evaluation.

Two methods (thesis Section 5.2):
  1. Scenario-based — 6 predefined weighting schemes (Table 4)
  2. Systematic perturbation — +/-30% on each criterion weight
"""

import json
import numpy as np
from pathlib import Path
from mcda_engine import (
    CRITERIA, BASELINE_WEIGHTS, rank_providers
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SCENARIOS = {
    "Balanced (Baseline)":  np.array([0.25, 0.30, 0.20, 0.15, 0.10]),
    "Cost-Focused":         np.array([0.15, 0.50, 0.15, 0.10, 0.10]),
    "Performance-Focused":  np.array([0.45, 0.20, 0.15, 0.10, 0.10]),
    "Security-Focused":     np.array([0.15, 0.20, 0.20, 0.35, 0.10]),
    "Enterprise":           np.array([0.20, 0.25, 0.25, 0.20, 0.10]),
    "Startup":              np.array([0.30, 0.40, 0.10, 0.10, 0.10]),
}


def run_scenarios():
    """Run all 6 predefined weighting scenarios."""
    out = {}
    for name, w in SCENARIOS.items():
        ranked = rank_providers(w.copy())
        out[name] = {
            "weights": dict(zip(CRITERIA, w.tolist())),
            "ranking": ranked,
            "winner": ranked[0]["provider"],
        }
    return out


def run_perturbation(variation=0.30):
    """
    Perturb each criterion weight by +/-30% from baseline.
    Other weights are rescaled so the total stays at 1.0.
    """
    out = {}
    for i, crit in enumerate(CRITERIA):
        for tag, factor in [("up", 1 + variation), ("down", 1 - variation)]:
            w = BASELINE_WEIGHTS.copy()
            w[i] *= factor
            rest = 1.0 - w[i]
            rest_base = 1.0 - BASELINE_WEIGHTS[i]
            for j in range(len(w)):
                if j != i:
                    w[j] = BASELINE_WEIGHTS[j] * rest / rest_base

            ranked = rank_providers(w)
            out[f"{crit}_{tag}"] = {
                "weights": dict(zip(CRITERIA, w.tolist())),
                "ranking": ranked,
                "winner": ranked[0]["provider"],
            }
    return out


def main():
    scenarios = run_scenarios()
    perturbations = run_perturbation()
    combined = {**scenarios, **perturbations}

    # Print summary table
    header = f"{'Scenario':<28}{'Winner':<8}{'GCP':<10}{'AWS':<10}{'Azure':<10}"
    print(header)
    print("-" * len(header))
    for name, res in combined.items():
        scores = {r["provider"]: r["score"] for r in res["ranking"]}
        print(f"{name:<28}{res['winner']:<8}"
              f"{scores.get('GCP', 0):<10.4f}"
              f"{scores.get('AWS', 0):<10.4f}"
              f"{scores.get('Azure', 0):<10.4f}")

    # Win count
    wins = {}
    for res in combined.values():
        p = res["winner"]
        wins[p] = wins.get(p, 0) + 1
    total = len(combined)
    print(f"\n{total} scenarios — "
          + ", ".join(f"{p}: {c}/{total}" for p, c in
                      sorted(wins.items(), key=lambda x: -x[1])))

    # Save full results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
