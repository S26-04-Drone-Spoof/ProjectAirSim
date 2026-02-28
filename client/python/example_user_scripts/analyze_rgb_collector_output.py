"""
Analyze RGB LiDAR Collector Output

Compares RGB collector data format with lidar_kush.py format
and shows how to use it with lidar_dataset_generator.py
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_single_scan(npz_path):
    """Analyze a single .npz scan file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {npz_path.name}")
    print(f"{'='*70}")
    
    data = np.load(npz_path)
    
    # Print all available fields
    print(f"\n📁 Available fields ({len(data.files)} total):")
    print("-" * 70)
    for field in sorted(data.files):
        arr = data[field]
        if isinstance(arr, np.ndarray):
            if arr.ndim == 0:
                print(f"  {field:25s} = {arr.item()}")
            elif arr.ndim == 1 and len(arr) <= 10:
                print(f"  {field:25s} shape={arr.shape} dtype={arr.dtype}")
                print(f"    └─> {arr}")
            else:
                print(f"  {field:25s} shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"  {field:25s} = {arr}")
    
    # Extract key data
    points = data['points']
    intensity = data['intensity']
    has_rgb = data['has_rgb']
    rgb = data['rgb']
    roughness = data['roughness']
    specular = data['specular']
    metallic = data['metallic']
    segmentation = data['segmentation']
    
    n_points = len(points)
    n_with_rgb = np.sum(has_rgb)
    pct_rgb = 100.0 * n_with_rgb / n_points
    
    print(f"\n📊 Point Cloud Statistics:")
    print("-" * 70)
    print(f"  Total points:           {n_points:,}")
    print(f"  Points with RGB:        {n_with_rgb:,} ({pct_rgb:.1f}%)")
    print(f"  Points without RGB:     {n_points - n_with_rgb:,} ({100-pct_rgb:.1f}%)")
    
    print(f"\n🎨 RGB Coverage Analysis:")
    print("-" * 70)
    if n_with_rgb > 0:
        rgb_valid = rgb[has_rgb]
        print(f"  RGB value range:        [{rgb_valid.min()}, {rgb_valid.max()}]")
        print(f"  Mean RGB:               {rgb_valid.mean(axis=0)}")
        
        # Analyze colors
        brightness = rgb_valid.mean(axis=1)
        print(f"  Brightness range:       [{brightness.min():.1f}, {brightness.max():.1f}]")
        print(f"  Mean brightness:        {brightness.mean():.1f}")
    else:
        print("  ⚠️  No RGB data in this scan!")
    
    print(f"\n⚡ Material Properties (RGB-derived):")
    print("-" * 70)
    if n_with_rgb > 0:
        print(f"  Intensity (with RGB):   [{intensity[has_rgb].min():.1f}, {intensity[has_rgb].max():.1f}]")
        print(f"  Roughness (with RGB):   [{roughness[has_rgb].min():.3f}, {roughness[has_rgb].max():.3f}]")
        print(f"  Specular (with RGB):    [{specular[has_rgb].min():.3f}, {specular[has_rgb].max():.3f}]")
        print(f"  Metallic (with RGB):    [{metallic[has_rgb].min():.3f}, {metallic[has_rgb].max():.3f}]")
        
        # Count non-zero material properties
        n_spec = np.sum(specular[has_rgb] > 0.1)
        n_metal = np.sum(metallic[has_rgb] > 0.1)
        print(f"\n  Specular points (>0.1): {n_spec} ({100*n_spec/n_with_rgb:.1f}% of RGB points)")
        print(f"  Metallic points (>0.1): {n_metal} ({100*n_metal/n_with_rgb:.1f}% of RGB points)")
    else:
        print(f"  Intensity (all zeros):  [{intensity.min():.1f}, {intensity.max():.1f}]")
        print("  ⚠️  No RGB data to analyze material properties")
    
    print(f"\n🏷️  Segmentation:")
    print("-" * 70)
    unique_seg = np.unique(segmentation)
    print(f"  Unique segment IDs:     {list(unique_seg)}")
    for seg_id in unique_seg:
        count = np.sum(segmentation == seg_id)
        print(f"    Segment {seg_id:3d}:          {count:,} points ({100*count/n_points:.1f}%)")
    
    return data


def compare_with_lidar_kush_format(npz_path):
    """Compare RGB collector format with lidar_kush.py format"""
    print(f"\n{'='*70}")
    print("📋 FORMAT COMPATIBILITY CHECK")
    print(f"{'='*70}")
    
    data = np.load(npz_path)
    
    # Expected fields from lidar_kush.py
    kush_fields = {
        'points', 'intensity', 'segmentation', 'timestamp', 'scan_id',
        'distance', 'normalized_intensity', 'roughness', 'backscatter',
        'intensity_variance', 'viewing_angle', 'has_intensity',
        'scene_name', 'drone_position', 'drone_velocity',
        'drone_orientation_quat', 'drone_orientation_euler'
    }
    
    # RGB collector fields
    rgb_fields = set(data.files)
    
    # Extra fields in RGB collector
    extra_fields = rgb_fields - kush_fields
    
    # Missing fields from lidar_kush
    missing_fields = kush_fields - rgb_fields
    
    # Common fields
    common_fields = kush_fields & rgb_fields
    
    print(f"\n✅ COMPATIBLE fields ({len(common_fields)} fields):")
    print("-" * 70)
    for field in sorted(common_fields):
        print(f"  ✓ {field}")
    
    print(f"\n⚠️  REPLACED fields (RGB uses different name/purpose):")
    print("-" * 70)
    if 'has_intensity' in missing_fields:
        print(f"  • has_intensity → has_rgb (same purpose, different name)")
        missing_fields.discard('has_intensity')
    
    if missing_fields:
        print(f"\n❌ MISSING fields ({len(missing_fields)} fields):")
        print("-" * 70)
        for field in sorted(missing_fields):
            print(f"  ✗ {field}")
    
    print(f"\n➕ ADDITIONAL fields in RGB collector ({len(extra_fields)} fields):")
    print("-" * 70)
    for field in sorted(extra_fields):
        arr = data[field]
        desc = ""
        if field == 'rgb':
            desc = " - RGB color values (N, 3)"
        elif field == 'specular':
            desc = " - Specular reflectance estimate [0-1]"
        elif field == 'metallic':
            desc = " - Metallic appearance estimate [0-1]"
        elif field == 'has_rgb':
            desc = " - Boolean mask (replaces has_intensity)"
        print(f"  + {field:20s} {desc}")
    
    print(f"\n{'='*70}")
    print("💡 INTEGRATION GUIDE:")
    print(f"{'='*70}")
    print("""
RGB Collector data is FULLY COMPATIBLE with lidar_kush.py format!

Key differences:
  1. 'intensity' comes from RGB brightness (not GPU LiDAR)
  2. 'has_rgb' replaces 'has_intensity' (same purpose)
  3. Additional fields: rgb, specular, metallic (bonus features!)

To use with lidar_dataset_generator.py:
  1. RGB collector saves to 'lidar_data_rgb/' folder
  2. lidar_kush.py saves to 'lidar_data/' folder
  3. Both use identical .npz format - interchange freely!
  
Example workflow:
  - Collect training data: Use RGB collector (easier, no GPU LiDAR tuning)
  - Load for AI model: Both formats work identically
  - Use extra fields: rgb, specular, metallic give more features
""")


def dataset_statistics(data_dir):
    """Analyze entire dataset"""
    print(f"\n{'='*70}")
    print(f"📦 DATASET STATISTICS: {data_dir.name}")
    print(f"{'='*70}")
    
    npz_files = sorted(data_dir.glob("scan_*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {data_dir}")
        return
    
    print(f"\n  Total scans: {len(npz_files)}")
    
    total_points = 0
    total_rgb_points = 0
    rgb_coverage_per_scan = []
    
    for npz_path in npz_files:
        data = np.load(npz_path)
        n_points = len(data['points'])
        n_rgb = np.sum(data['has_rgb'])
        total_points += n_points
        total_rgb_points += n_rgb
        if n_points > 0:
            rgb_coverage_per_scan.append(100.0 * n_rgb / n_points)
    
    avg_coverage = 100.0 * total_rgb_points / total_points if total_points > 0 else 0
    
    print(f"\n  Total points:       {total_points:,}")
    print(f"  Total RGB points:   {total_rgb_points:,}")
    print(f"  Overall coverage:   {avg_coverage:.1f}%")
    
    if rgb_coverage_per_scan:
        print(f"\n  Per-scan RGB coverage:")
        print(f"    Min:  {min(rgb_coverage_per_scan):.1f}%")
        print(f"    Max:  {max(rgb_coverage_per_scan):.1f}%")
        print(f"    Mean: {np.mean(rgb_coverage_per_scan):.1f}%")
        print(f"    Std:  {np.std(rgb_coverage_per_scan):.1f}%")
    
    # Distribution of coverage
    scans_0_10 = sum(1 for c in rgb_coverage_per_scan if c < 10)
    scans_10_30 = sum(1 for c in rgb_coverage_per_scan if 10 <= c < 30)
    scans_30_50 = sum(1 for c in rgb_coverage_per_scan if 30 <= c < 50)
    scans_50_plus = sum(1 for c in rgb_coverage_per_scan if c >= 50)
    
    print(f"\n  Coverage distribution:")
    print(f"    0-10%:   {scans_0_10:3d} scans ({100*scans_0_10/len(npz_files):.1f}%)")
    print(f"    10-30%:  {scans_10_30:3d} scans ({100*scans_10_30/len(npz_files):.1f}%)")
    print(f"    30-50%:  {scans_30_50:3d} scans ({100*scans_30_50/len(npz_files):.1f}%)")
    print(f"    50%+:    {scans_50_plus:3d} scans ({100*scans_50_plus/len(npz_files):.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rgb_coverage_per_scan, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('RGB Coverage (%)')
    plt.ylabel('Number of Scans')
    plt.title(f'RGB Coverage Distribution ({len(npz_files)} scans)')
    plt.axvline(avg_coverage, color='red', linestyle='--', label=f'Mean: {avg_coverage:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = data_dir / "rgb_coverage_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Saved histogram: {output_path}")


def main():
    """Main analysis"""
    data_dir = Path("lidar_data_rgb")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run rgb_lidar_collector.py first!")
        return
    
    npz_files = sorted(data_dir.glob("scan_*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {data_dir}")
        return
    
    print(f"\n🔍 Found {len(npz_files)} scan files")
    
    # Analyze first scan in detail
    first_scan = npz_files[0]
    data = analyze_single_scan(first_scan)
    
    # Compare format
    compare_with_lidar_kush_format(first_scan)
    
    # Dataset statistics
    dataset_statistics(data_dir)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"""
Next steps:
  1. Review the statistics above
  2. Check the histogram: {data_dir}/rgb_coverage_histogram.png
  3. Use this data with lidar_dataset_generator.py workflow
  4. Access extra features: data['rgb'], data['specular'], data['metallic']
  
Example code to load and use:
  
  import numpy as np
  data = np.load('lidar_data_rgb/scan_0000.npz')
  
  # Standard fields (compatible with lidar_kush.py)
  points = data['points']              # (N, 3) XYZ coordinates
  intensity = data['intensity']        # (N,) RGB-derived intensity
  roughness = data['roughness']        # (N,) Surface roughness
  segmentation = data['segmentation']  # (N,) Object segments
  
  # Extra RGB features
  rgb = data['rgb']                    # (N, 3) RGB colors
  specular = data['specular']          # (N,) Specular estimate
  metallic = data['metallic']          # (N,) Metallic estimate
  has_rgb = data['has_rgb']            # (N,) Boolean mask
  
  # Filter to RGB points only
  rgb_points = points[has_rgb]
  rgb_colors = rgb[has_rgb]
""")


if __name__ == "__main__":
    main()
