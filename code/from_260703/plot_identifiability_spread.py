"""
plot_identifiability_spread.py

Adds spread visualization to the identifiability (true vs. peak estimate)
results:
  1. Per-case scatter with a 1-SD (or chosen confidence level) covariance
     ellipse around the cluster of peak estimates, true value marked as a star.
  2. Two grid-level calibration plots (true vs. mean estimate, with error
     bars = SD across seeds, and a y=x reference line) -- one for R0, one
     for sigma.

Requires the per-case peak_estimates_*.csv files (TRUE_R0, TRUE_SIGMA,
peak_R0, peak_sigma; one row per seed), e.g. the output already produced
by your flow_process_sigma*_R0*.py scripts.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
input_dir = "../../experimental_data/from_260703/"
file_pattern = "peak_estimates_*.csv"
output_dir = "../../figures/from_260703/dentifiability_figures/"
confidence_level = 0.95  # for ellipses; use 1.0 for raw 1-SD ellipse (n_std=1)


def load_all_cases(input_dir, file_pattern):
    files = sorted(glob.glob(str(Path(input_dir) / file_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in '{input_dir}'.")
    cases = {}
    for f in files:
        df = pd.read_csv(f)
        true_R0 = df["TRUE_R0"].iloc[0]
        true_sigma = df["TRUE_SIGMA"].iloc[0]
        cases[(true_R0, true_sigma)] = df
    return cases


def confidence_ellipse(x, y, ax, n_std=2.4477, **kwargs):
    """
    Draw a covariance confidence ellipse for the (x, y) points onto ax.
    n_std=2.4477 corresponds to a 95% confidence ellipse for 2D Gaussian data
    (chi-square with 2 dof); use n_std=1 for a raw 1-SD ellipse.
    """
    if len(x) < 2:
        return None  # can't estimate covariance from a single point

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Eigen-decomposition-free approach via Pearson correlation (standard mpl recipe)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        facecolor=kwargs.pop("facecolor", "none"),
        edgecolor=kwargs.pop("edgecolor", "blue"),
        alpha=kwargs.pop("alpha", 0.3),
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_case_with_ellipse(true_R0, true_sigma, df, output_dir,
                            xlim=(1, 4), ylim=(0.2, 1.0)):
    fig, ax = plt.subplots(figsize=(6, 5))

    x = df["peak_R0"].to_numpy()
    y = df["peak_sigma"].to_numpy()

    confidence_ellipse(x, y, ax, n_std=2.4477,
                        facecolor="steelblue", edgecolor="steelblue", alpha=0.15)
    confidence_ellipse(x, y, ax, n_std=1.0,
                        facecolor="none", edgecolor="steelblue", alpha=0.6, linestyle="--")

    ax.scatter(x, y, marker="+", color="blue", s=150, zorder=3, label="Peak estimates (per seed)")
    ax.scatter(true_R0, true_sigma, marker="*", color="red", s=300, zorder=4, label="True value")
    ax.scatter(x.mean(), y.mean(), marker="x", color="black", s=150, zorder=4, label="Mean estimate")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$R_0$", fontsize=13)
    ax.set_ylabel(r"$\sigma$", fontsize=13)
    ax.set_title(f"True (R0={true_R0}, $\\sigma$={true_sigma}) vs. estimates\n"
                f"(dashed = 1 SD, shaded = 95% ellipse, n={len(x)})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{output_dir}/ellipse_sigma{true_sigma}_R0{true_R0}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


def calibration_plot(cases, output_dir, param="R0", xlim=(1, 4)):
    """Grid-level calibration plot: true vs. mean estimate with SD error bars."""
    rows = []
    for (true_R0, true_sigma), df in cases.items():
        true_val = true_R0 if param == "R0" else true_sigma
        est = df["peak_R0"].to_numpy() if param == "R0" else df["peak_sigma"].to_numpy()
        rows.append({
            "true": true_val,
            "mean_est": est.mean(),
            "sd_est": est.std(ddof=1) if len(est) > 1 else 0.0,
        })
    summary = pd.DataFrame(rows).sort_values("true")

    fig, ax = plt.subplots(figsize=(6, 6))
    lims = xlim if param == "R0" else (0.2, 1.0)
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y = x (perfect estimation)")
    ax.errorbar(summary["true"], summary["mean_est"], yerr=summary["sd_est"],
                fmt="o", color="steelblue", ecolor="steelblue", alpha=0.7,
                capsize=3, markersize=5, label="Mean estimate \u00b1 1 SD")

    label = r"$R_0$" if param == "R0" else r"$\sigma$"
    ax.set_xlabel(f"True {label}", fontsize=13)
    ax.set_ylabel(f"Estimated {label} (mean across seeds)", fontsize=13)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"Calibration: {label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{output_dir}/calibration_{param}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


def main():
    cases = load_all_cases(input_dir, file_pattern)
    print(f"Loaded {len(cases)} validation combinations.")

    print("Generating per-case ellipse plots...")
    for (true_R0, true_sigma), df in cases.items():
        fname = plot_case_with_ellipse(true_R0, true_sigma, df, output_dir)
        print(f"  Saved {fname}")

    print("Generating grid-level calibration plots...")
    fname_r0 = calibration_plot(cases, output_dir, param="R0")
    fname_sigma = calibration_plot(cases, output_dir, param="sigma")
    print(f"  Saved {fname_r0}")
    print(f"  Saved {fname_sigma}")


if __name__ == "__main__":
    main()
