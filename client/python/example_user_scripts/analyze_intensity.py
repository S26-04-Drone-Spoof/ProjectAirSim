"""
Intensity Analysis Script
Loads saved LiDAR scans and checks whether intensity varies across surfaces.
Run this AFTER lidar_kush.py has collected data into lidar_data_test/.

Usage:
    python analyze_intensity.py [data_dir] [max_distance_m]

    data_dir        defaults to "lidar_data_test"
    max_distance_m  defaults to 500  (filters out background/sky pixels)
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── helpers ───────────────────────────────────────────────────────────────────

MAX_VALID_DISTANCE = 500.0  # metres — anything beyond this is a background pixel


def load_scans(data_dir: Path):
    files = sorted(data_dir.glob("scan_*.npz"))
    if not files:
        raise FileNotFoundError(f"No scan_*.npz files found in {data_dir}")
    print(f"Found {len(files)} scan file(s) in {data_dir}")

    all_points    = []
    all_intensity = []
    all_distance  = []
    all_norm_int  = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        pts  = d["points"]                   # (N, 3)
        inty = d["intensity"]                # (N,)
        dist = d["distance"]                 # (N,)
        norm = d["normalized_intensity"]     # (N,)

        all_points.append(pts)
        all_intensity.append(inty)
        all_distance.append(dist)
        all_norm_int.append(norm)

    points    = np.concatenate(all_points,    axis=0)
    intensity = np.concatenate(all_intensity, axis=0)
    distance  = np.concatenate(all_distance,  axis=0)
    norm_int  = np.concatenate(all_norm_int,  axis=0)

    total_raw = len(intensity)
    print(f"Total points loaded (raw): {total_raw:,}")

    # ── Filter out invalid / background points ─────────────────────────────
    # GPU lidar has two sources of garbage points:
    #   1. Background/sky pixels → astronomically large world coords (distance > MAX_VALID_DISTANCE)
    #   2. Shader threads outside the 45°-135° arc → zero-initialized buffer → (0,0,0) points
    #      These get distance clamped to 0.1 m and have intensity = 0.
    #
    # Filter (0,0,0) points by requiring at least one non-zero coordinate.
    MIN_VALID_DISTANCE = 0.5   # metres — below this is a (0,0,0) zero-init point
    non_zero = np.any(points != 0.0, axis=1)
    valid = (
        non_zero &
        np.isfinite(distance) &
        (distance > MIN_VALID_DISTANCE) &
        (distance < MAX_VALID_DISTANCE) &
        np.isfinite(intensity) &
        np.all(np.isfinite(points), axis=1)
    )

    n_dropped = total_raw - valid.sum()
    n_zero    = (~non_zero).sum()
    if n_dropped:
        pct = 100.0 * n_dropped / total_raw
        print(f"Dropped {n_dropped:,} invalid points ({pct:.1f}%):")
        print(f"  {n_zero:,} zero-origin points (unprocessed shader threads outside 45°-135° arc)")
        print(f"  {n_dropped - n_zero:,} background/sky pixels (distance outside [{MIN_VALID_DISTANCE}, {MAX_VALID_DISTANCE}] m)")

    points    = points[valid]
    intensity = intensity[valid]
    distance  = distance[valid]
    norm_int  = norm_int[valid]

    # Clip any remaining inf/nan in norm_int (can arise from intensity*dist² overflow)
    norm_int = np.where(np.isfinite(norm_int), norm_int, 0.0).astype(np.float32)

    if len(intensity) == 0:
        raise ValueError(
            "No valid points remain after filtering! "
            "All points appear to be background pixels or have non-finite values. "
            "Check GPU lidar config (vertical-fov, laser-range) or try a larger max_distance_m."
        )

    print(f"Valid points after filtering: {len(intensity):,}")
    return points, intensity, distance, norm_int


def print_stats(label, arr):
    if len(arr) == 0:
        print(f"  {label:30s}  (no data)")
        return
    # Use scientific notation so tiny values don't all print as 0.0000
    print(f"  {label:30s}  min={arr.min():.6g}  max={arr.max():.6g}  "
          f"mean={arr.mean():.6g}  std={arr.std():.6g}  "
          f"median={np.median(arr):.6g}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_intensity_histogram(intensity, norm_int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Intensity Distributions", fontsize=13)

    axes[0].hist(intensity, bins=100, color="steelblue", edgecolor="none")
    axes[0].set_title("Raw Intensity")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Point count")

    # Use only finite norm_int values for the histogram
    finite_norm = norm_int[np.isfinite(norm_int)]
    if len(finite_norm) > 0:
        axes[1].hist(finite_norm, bins=100, color="darkorange", edgecolor="none")
    else:
        axes[1].text(0.5, 0.5, "No finite values", ha="center", va="center",
                     transform=axes[1].transAxes)
    axes[1].set_title("Normalized Intensity  (raw × distance²)")
    axes[1].set_xlabel("Normalized intensity")
    axes[1].set_ylabel("Point count")

    plt.tight_layout()


def plot_intensity_vs_distance(intensity, distance):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distance, intensity, s=1, alpha=0.3, color="steelblue")
    ax.set_title("Raw Intensity vs Distance\n"
                 "(flat line → distance-only formula, clusters → material variation)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Raw intensity")
    plt.tight_layout()


def plot_3d_intensity(points, intensity):
    """3-D scatter coloured by intensity – lets you see if certain surfaces are brighter."""
    MAX_POINTS = 20_000
    if len(intensity) > MAX_POINTS:
        idx = np.random.choice(len(intensity), MAX_POINTS, replace=False)
        pts = points[idx]
        inty = intensity[idx]
    else:
        pts = points
        inty = intensity

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=inty, cmap="plasma", s=1, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Raw intensity", shrink=0.6)
    ax.set_title("Point Cloud coloured by Intensity\n"
                 "(bright spots on high-specular surfaces?)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.tight_layout()


def plot_intensity_percentiles(intensity):
    """Shows how spread the intensities are – large spread suggests material variation."""
    percentiles = np.arange(0, 101, 5)
    values = np.percentile(intensity, percentiles)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(percentiles, values, marker="o", color="steelblue")
    ax.set_title("Intensity Percentile Plot\n"
                 "(steep curve at top = bright outliers from reflective surfaces)")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Raw intensity")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()


def cluster_by_intensity(intensity, label="Intensity", n_bins=5):
    """
    Naively bin points by intensity and report how many fall in each bin.
    If all points clump in one bin → intensity doesn't vary.
    """
    lo, hi = intensity.min(), intensity.max()
    if lo == hi:
        print(f"\n  {label} bins: all {len(intensity):,} points have value {lo:.4g} "
              "(no variation — intensity may not be computed by this lidar type)")
        return

    bins = np.linspace(lo, hi, n_bins + 1)
    labels = [f"[{bins[i]:.3e}, {bins[i+1]:.3e})" for i in range(n_bins)]
    counts, _ = np.histogram(intensity, bins=bins)
    total = len(intensity)

    print(f"\n  {label} bins (equal-width):")
    for lbl, cnt in zip(labels, counts):
        bar = "#" * int(50 * cnt / total)
        print(f"    {lbl}  {cnt:7,}  ({100*cnt/total:5.1f}%)  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lidar_data_test")
    global MAX_VALID_DISTANCE
    if len(sys.argv) > 2:
        MAX_VALID_DISTANCE = float(sys.argv[2])

    if not data_dir.exists():
        print(f"ERROR: directory '{data_dir}' not found.")
        print("Run lidar_kush.py first to collect data, then re-run this script.")
        sys.exit(1)

    points, intensity, distance, norm_int = load_scans(data_dir)

    print("\n=== Summary statistics ===")
    print_stats("raw intensity",        intensity)
    print_stats("distance (m)",         distance)
    print_stats("normalized intensity", norm_int)

    cluster_by_intensity(intensity, label="Raw intensity")
    cluster_by_intensity(norm_int,  label="Normalized intensity (raw x dist^2)")

    # ── verdict ───────────────────────────────────────────────────────────────
    spread  = float(intensity.max() - intensity.min())
    mean_i  = float(intensity.mean())
    rel_std = float(intensity.std()) / (mean_i + 1e-9)

    finite_norm = norm_int[np.isfinite(norm_int) & (norm_int > 0)]
    norm_rel_std = (float(finite_norm.std()) / (float(finite_norm.mean()) + 1e-9)
                    if len(finite_norm) > 0 else 0.0)

    print(f"\n=== Verdict ===")
    print(f"  Raw intensity range   : {spread:.4g}  (mean={mean_i:.4g}, rel_std={rel_std:.3f})")
    print(f"  Norm intensity range  : {float(finite_norm.min()):.4g} – {float(finite_norm.max()):.4g}"
          f"  (mean={float(finite_norm.mean()):.4g}, rel_std={norm_rel_std:.3f})")

    if spread == 0.0:
        print("  → Intensity is identically zero — the shader is not writing intensity data.")
        print("    Check GPU lidar shader (LidarIntensityShader.usf) G-buffer reads.")
    elif mean_i < 1e-4 and norm_rel_std > 0.30:
        print(f"  → Raw intensity tiny (mean={mean_i:.3e}) but NORMALIZED intensity varies strongly")
        print(f"    (rel_std={norm_rel_std:.2f}).  Surface differences ARE being detected.")
        print(f"    Absolute scale is low because scene materials have low default Specular (~0.04).")
        print(f"    Use normalized_intensity as your signal, not raw intensity.")
    elif mean_i < 1e-4:
        print(f"  → Intensity is extremely small (mean ≈ {mean_i:.3e}) and barely varies.")
        print("    The G-buffer IS being read, but values are near-zero.")
        print("    Check material Specular/Roughness/Metallic values in the scene.")
    elif rel_std > 0.10:
        print("  → Intensities DO vary — lidar is picking up surface differences.")
    else:
        print("  → Intensities are nearly uniform — lidar is NOT picking up material.")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_intensity_histogram(intensity, norm_int)
    plot_intensity_vs_distance(intensity, distance)
    plot_3d_intensity(points, intensity)
    plot_intensity_percentiles(intensity)
    plt.show()


if __name__ == "__main__":
    main()
