"""
aggregate_identifiability_results.py

Aggregates the per-case peak_estimates_sigma{S}_R0{R}.csv files (one file per
(R0_true, sigma_true) combination, each containing one row per seed) into:
  1. A summary CSV (one row per combination) with mean/bias/SD for R0 and sigma.
  2. A LaTeX longtable (Table S1) ready to paste into the SI Appendix.

Usage:
    Place this script in the same directory as your peak_estimates_*.csv files
    (or set `input_dir` below), then run:
        python aggregate_identifiability_results.py
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
input_dir = "../../experimental_data/from_260703/"  # directory containing peak_estimates_*.csv files
file_pattern = "peak_estimates_*.csv"
output_csv = "../../experimental_data/from_260703/dentifiability_data/identifiability_summary.csv"
output_tex = "../../experimental_data/from_260703/dentifiability_data/table_s1_identifiability.tex"
output_compact_csv = "../../experimental_data/from_260703/dentifiability_data/identifiability_summary_compact.csv"
output_compact_tex = "../../experimental_data/from_260703/dentifiability_data/table_identifiability_compact.tex"


def load_all_cases(input_dir, file_pattern):
    files = sorted(glob.glob(str(Path(input_dir) / file_pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in '{input_dir}'. "
            "Check the path / pattern."
        )

    rows = []
    for f in files:
        df = pd.read_csv(f)
        # Expect columns: TRUE_R0, TRUE_SIGMA, peak_R0, peak_sigma
        true_R0 = df["TRUE_R0"].iloc[0]
        true_sigma = df["TRUE_SIGMA"].iloc[0]
        n_seeds = len(df)

        r0_est = df["peak_R0"].to_numpy()
        sigma_est = df["peak_sigma"].to_numpy()

        rows.append({
            "R0_true": true_R0,
            "sigma_true": true_sigma,
            "n_seeds": n_seeds,
            "R0_mean": r0_est.mean(),
            "R0_bias": r0_est.mean() - true_R0,
            "R0_sd": r0_est.std(ddof=1) if n_seeds > 1 else np.nan,
            "sigma_mean": sigma_est.mean(),
            "sigma_bias": sigma_est.mean() - true_sigma,
            "sigma_sd": sigma_est.std(ddof=1) if n_seeds > 1 else np.nan,
            "source_file": Path(f).name,
        })

    summary = pd.DataFrame(rows).sort_values(["sigma_true", "R0_true"]).reset_index(drop=True)
    return summary


def write_latex_longtable(summary, output_tex):
    """Write a longtable suitable for an SI Appendix (Table S1, full 70 rows)."""
    lines = []
    lines.append(r"\begin{longtable}{ccccccccc}")
    lines.append(r"\caption{Practical identifiability analysis: estimation accuracy across the "
                 r"full validation grid (70 true $(R_0,\sigma)$ combinations, $n$ seeds each). "
                 r"Bias is mean estimate minus true value; SD is the standard deviation of "
                 r"the per-seed estimates.} \label{tab:s1_identifiability} \\")
    lines.append(r"\hline")
    header = (r"$R_0^{\text{true}}$ & $\sigma^{\text{true}}$ & $n$ & "
              r"$\overline{R_0}$ & Bias $R_0$ & SD $R_0$ & "
              r"$\overline{\sigma}$ & Bias $\sigma$ & SD $\sigma$ \\")
    lines.append(header)
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")
    lines.append(r"\hline")
    lines.append(header)
    lines.append(r"\hline")
    lines.append(r"\endhead")
    lines.append(r"\hline \multicolumn{9}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")

    for _, row in summary.iterrows():
        lines.append(
            f"{row['R0_true']:.2f} & {row['sigma_true']:.3f} & {int(row['n_seeds'])} & "
            f"{row['R0_mean']:.3f} & {row['R0_bias']:+.3f} & {row['R0_sd']:.3f} & "
            f"{row['sigma_mean']:.3f} & {row['sigma_bias']:+.3f} & {row['sigma_sd']:.3f} \\\\"
        )

    lines.append(r"\end{longtable}")

    Path(output_tex).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote LaTeX longtable (full, 70 rows) to: {output_tex}")


def build_compact_summary(summary):
    """
    Collapse the full 70-row grid to one row per true sigma value, averaging
    bias and SD across the 10 R0 values at that sigma. Useful as a quick,
    main-text-friendly companion table to the full SI longtable.
    """
    compact = (
        summary
        .groupby("sigma_true")
        .agg(
            n_R0_values=("R0_true", "nunique"),
            R0_bias_mean=("R0_bias", "mean"),
            R0_bias_sd=("R0_bias", "std"),
            R0_sd_mean=("R0_sd", "mean"),
            sigma_bias_mean=("sigma_bias", "mean"),
            sigma_bias_sd=("sigma_bias", "std"),
            sigma_sd_mean=("sigma_sd", "mean"),
        )
        .reset_index()
        .sort_values("sigma_true")
    )
    return compact


def write_latex_compact_table(compact, output_tex):
    """Write a regular (non-long) table: one row per sigma_true, averaged over R0."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Identifiability analysis summary: estimation bias and spread, "
                 r"averaged across the 10 true $R_0$ values at each true $\sigma$ "
                 r"(see Table S1 for the full 70-combination breakdown). "
                 r"Bias values are mean $\pm$ between-$R_0$ SD of the per-combination bias; "
                 r"SD columns are the average within-combination (across-seed) SD.}")
    lines.append(r"\label{tab:identifiability_compact}")
    lines.append(r"\begin{tabular}{cccccc}")
    lines.append(r"\hline")
    lines.append(r"$\sigma^{\text{true}}$ & $n$ ($R_0$ values) & "
                 r"Bias $R_0$ & SD $R_0$ (within-seed) & "
                 r"Bias $\sigma$ & SD $\sigma$ (within-seed) \\")
    lines.append(r"\hline")
    for _, row in compact.iterrows():
        lines.append(
            f"{row['sigma_true']:.3f} & {int(row['n_R0_values'])} & "
            f"{row['R0_bias_mean']:+.3f} $\\pm$ {row['R0_bias_sd']:.3f} & "
            f"{row['R0_sd_mean']:.3f} & "
            f"{row['sigma_bias_mean']:+.3f} $\\pm$ {row['sigma_bias_sd']:.3f} & "
            f"{row['sigma_sd_mean']:.3f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    Path(output_tex).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote LaTeX compact table to: {output_tex}")


def main():
    summary = load_all_cases(input_dir, file_pattern)
    summary.to_csv(output_csv, index=False)
    print(f"Wrote summary CSV to: {output_csv}")
    print(f"  {len(summary)} combinations found "
          f"(expected 70 = 10 R0 values x 7 sigma values)")

    missing = 70 - len(summary)
    if missing != 0:
        print(f"  WARNING: {missing} combination(s) missing or unexpected count found. "
              f"Check file_pattern / input_dir.")

    write_latex_longtable(summary, output_tex)

    compact = build_compact_summary(summary)
    compact.to_csv(output_compact_csv, index=False)
    print(f"Wrote compact summary CSV to: {output_compact_csv}")
    write_latex_compact_table(compact, output_compact_tex)

    # Quick console summary to eyeball the pattern before pasting into the paper
    print("\nBias summary by true sigma (averaged across R0):")
    print(summary.groupby("sigma_true")[["R0_bias", "sigma_bias"]].mean().round(3))


if __name__ == "__main__":
    main()
