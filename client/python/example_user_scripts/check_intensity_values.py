#!/usr/bin/env python3
"""Quick check of intensity values in the data"""

import numpy as np
from pathlib import Path

# Load first scan
scan_file = Path("lidar_data/scan_0000.npz")
if not scan_file.exists():
    print("No scan_0000.npz found. Run lidar_kush.py first.")
    exit()

data = np.load(scan_file)
points = data['points']
intensity = data['intensity']

# Compute distances
distance = np.linalg.norm(points, axis=1)

# Compute normalized
distance[distance < 0.1] = 0.1
normalized = intensity * (distance ** 2)

print("="*60)
print("INTENSITY VALUE CHECK - Scan 0")
print("="*60)

print("\nRAW INTENSITY:")
print(f"  Min:    {np.min(intensity):.6f}")
print(f"  Max:    {np.max(intensity):.6f}")
print(f"  Mean:   {np.mean(intensity):.6f}")
print(f"  Median: {np.median(intensity):.6f}")

print("\nDISTANCES:")
print(f"  Min:    {np.min(distance):.2f} m")
print(f"  Max:    {np.max(distance):.2f} m")
print(f"  Mean:   {np.mean(distance):.2f} m")
print(f"  Median: {np.median(distance):.2f} m")

print("\nNORMALIZED INTENSITY (distance² corrected):")
print(f"  Min:    {np.min(normalized):.6f}")
print(f"  Max:    {np.max(normalized):.6f}")
print(f"  Mean:   {np.mean(normalized):.6f}")
print(f"  Median: {np.median(normalized):.6f}")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)

if np.max(intensity) < 0.01:
    print("⚠️  Raw intensity is VERY low (< 0.01)")
    print("   LiDAR might not be configured to return intensity")
elif np.max(intensity) > 0:
    print("✅ Raw intensity looks good (0-{:.2f} range)".format(np.max(intensity)))

if np.max(normalized) < 1:
    print("⚠️  Normalized intensity is VERY low (< 1)")
    print("   This might cause visualization issues")
else:
    print("✅ Normalized intensity range: 0-{:.1f}".format(np.max(normalized)))

print("\nNormalization factor (distance²):")
print(f"  Min: {np.min(distance**2):.2f}x")
print(f"  Max: {np.max(distance**2):.2f}x")
print(f"  Mean: {np.mean(distance**2):.2f}x")
