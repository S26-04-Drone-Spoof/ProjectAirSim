#!/usr/bin/env python3
"""
Verify LiDAR data distributions and compute detailed statistics
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_all_scans(data_dir="lidar_data", max_scans=None):
    """Load scan files"""
    data_dir = Path(data_dir)
    scan_files = sorted(data_dir.glob("scan_*.npz"))

    if max_scans:
        scan_files = scan_files[:max_scans]

    print(f"Loading {len(scan_files)} scans...")

    all_intensity = []
    all_relative_reflectivity = []

    for i, scan_file in enumerate(scan_files):
        if i % 500 == 0:
            print(f"  Processing scan {i}/{len(scan_files)}...")

        data = np.load(scan_file)
        points = data['points']
        intensity = data['intensity']

        # Compute distance
        distance = np.linalg.norm(points, axis=1)
        distance = np.maximum(distance, 0.1)  # Avoid division by zero

        # Compute relative reflectivity (distance-normalized intensity)
        relative_reflectivity = intensity * (distance ** 2)

        all_intensity.extend(intensity.flatten())
        all_relative_reflectivity.extend(relative_reflectivity.flatten())

    return np.array(all_intensity), np.array(all_relative_reflectivity)


print("="*70)
print("LIDAR DATA DISTRIBUTION VERIFICATION")
print("="*70)

# Load data
intensity, relative_reflectivity = load_all_scans(max_scans=200)  # First 200 scans

print("\n" + "="*70)
print("RAW INTENSITY STATISTICS")
print("="*70)

# Filter out invalid values
intensity_valid = intensity[np.isfinite(intensity)]
print(f"\nTotal points: {len(intensity):,}")
print(f"Valid points: {len(intensity_valid):,} ({100*len(intensity_valid)/len(intensity):.2f}%)")
print(f"Invalid points: {len(intensity) - len(intensity_valid):,}")

print(f"\nRaw Intensity:")
print(f"  Min:        {np.min(intensity_valid):.10f}")
print(f"  Max:        {np.max(intensity_valid):.10f}")
print(f"  Mean:       {np.mean(intensity_valid):.10f}")
print(f"  Median:     {np.median(intensity_valid):.10f}")
print(f"  Std Dev:    {np.std(intensity_valid):.10f}")

percentiles = np.percentile(intensity_valid, [1, 5, 25, 50, 75, 95, 99])
print(f"\nPercentiles:")
print(f"  1%:   {percentiles[0]:.10f}")
print(f"  5%:   {percentiles[1]:.10f}")
print(f"  25%:  {percentiles[2]:.10f}")
print(f"  50%:  {percentiles[3]:.10f}")
print(f"  75%:  {percentiles[4]:.10f}")
print(f"  95%:  {percentiles[5]:.10f}")
print(f"  99%:  {percentiles[6]:.10f}")

# Count zero and near-zero values
zero_count = np.sum(intensity_valid == 0)
near_zero_count = np.sum((intensity_valid > 0) & (intensity_valid < 1e-6))
print(f"\nValue distribution:")
print(f"  Zero values:       {zero_count:,} ({100*zero_count/len(intensity_valid):.2f}%)")
print(f"  Near-zero (< 1e-6): {near_zero_count:,} ({100*near_zero_count/len(intensity_valid):.2f}%)")
print(f"  Valid (≥ 1e-6):    {len(intensity_valid) - zero_count - near_zero_count:,}")

print("\n" + "="*70)
print("RELATIVE REFLECTIVITY STATISTICS (Distance² Normalized)")
print("="*70)

# Filter out invalid values
reflectivity_valid = relative_reflectivity[np.isfinite(relative_reflectivity)]
print(f"\nTotal points: {len(relative_reflectivity):,}")
print(f"Valid points: {len(reflectivity_valid):,} ({100*len(reflectivity_valid)/len(relative_reflectivity):.2f}%)")
print(f"Invalid points: {len(relative_reflectivity) - len(reflectivity_valid):,}")

print(f"\nRelative Reflectivity:")
print(f"  Min:        {np.min(reflectivity_valid):.10f}")
print(f"  Max:        {np.max(reflectivity_valid):.10f}")
print(f"  Mean:       {np.mean(reflectivity_valid):.10f}")
print(f"  Median:     {np.median(reflectivity_valid):.10f}")
print(f"  Std Dev:    {np.std(reflectivity_valid):.10f}")

percentiles_refl = np.percentile(reflectivity_valid, [1, 5, 25, 50, 75, 95, 99])
print(f"\nPercentiles:")
print(f"  1%:   {percentiles_refl[0]:.10f}")
print(f"  5%:   {percentiles_refl[1]:.10f}")
print(f"  25%:  {percentiles_refl[2]:.10f}")
print(f"  50%:  {percentiles_refl[3]:.10f}")
print(f"  75%:  {percentiles_refl[4]:.10f}")
print(f"  95%:  {percentiles_refl[5]:.10f}")
print(f"  99%:  {percentiles_refl[6]:.10f}")

# Check for negative values
negative_count = np.sum(reflectivity_valid < 0)
positive_count = np.sum(reflectivity_valid > 0)
zero_refl_count = np.sum(reflectivity_valid == 0)

print(f"\nValue distribution:")
print(f"  Negative values: {negative_count:,} ({100*negative_count/len(reflectivity_valid):.2f}%)")
print(f"  Zero values:     {zero_refl_count:,} ({100*zero_refl_count/len(reflectivity_valid):.2f}%)")
print(f"  Positive values: {positive_count:,} ({100*positive_count/len(reflectivity_valid):.2f}%)")

print("\n" + "="*70)
print("DATA QUALITY ASSESSMENT")
print("="*70)

# Assess data quality
print("\n✅ Raw Intensity:")
if np.max(intensity_valid) > 0:
    print(f"   - Values are present (max = {np.max(intensity_valid):.6f})")
else:
    print("   ⚠️  All values are zero!")

if np.std(intensity_valid) > 0:
    print(f"   - Has variation (std = {np.std(intensity_valid):.6f})")
else:
    print("   ⚠️  No variation in values!")

print("\n✅ Relative Reflectivity:")
if negative_count > 0:
    print(f"   ⚠️  WARNING: {negative_count:,} negative values detected!")
    print("   Physical reflectivity cannot be negative.")
    print("   This suggests a computation error or data issue.")
else:
    print("   - All values are non-negative (physically valid)")

if np.max(reflectivity_valid) > 0:
    print(f"   - Values are present (max = {np.max(reflectivity_valid):.6f})")
else:
    print("   ⚠️  All values are zero!")

# Check if histogram would be visible
refl_range = np.max(reflectivity_valid) - np.min(reflectivity_valid)
print(f"\n✅ Distribution Visibility:")
print(f"   - Value range: {refl_range:.10f}")
if refl_range < 0.001:
    print("   ⚠️  Range is very small (< 0.001)")
    print("   This explains why the histogram appears empty.")
    print("   Values are too concentrated for standard binning.")
else:
    print(f"   - Range is reasonable for histogram visualization")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if negative_count > 0:
    print("\n⚠️  ISSUE 1: Negative reflectivity values")
    print("   FIX: Check the normalization formula")
    print("   Expected: reflectivity = intensity * distance²")
    print("   Ensure intensity values are non-negative")

if refl_range < 0.001:
    print("\n⚠️  ISSUE 2: Very narrow value range in relative reflectivity")
    print("   FIX: Use log scale or adjust histogram bins")
    print("   Example: plt.hist(data, bins=100)")
    print("   Or: Use log scale: plt.yscale('log')")

if np.mean(intensity_valid) < 0.01:
    print("\n⚠️  ISSUE 3: Very low raw intensity values")
    print("   This might indicate:")
    print("   - LiDAR intensity not properly configured")
    print("   - Non-reflective surfaces (dark/absorptive materials)")
    print("   - Sensor calibration needed")

print("\n" + "="*70)
