"""
Example: Loading RGB LiDAR Data for AI Training

Shows how to load and preprocess RGB LiDAR collector data
for machine learning models. Compatible with both:
  - lidar_kush.py output (GPU LiDAR)
  - rgb_lidar_collector.py output (RGB-derived)
"""

import numpy as np
from pathlib import Path


def load_scan(npz_path):
    """
    Load a single scan from .npz file.
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        Dictionary with all scan data
    """
    data = np.load(npz_path)
    
    return {
        # Geometry
        'points': data['points'],              # (N, 3) XYZ coordinates
        'distance': data['distance'],          # (N,) Distance from sensor
        
        # Material properties
        'intensity': data['intensity'],        # (N,) Reflectance intensity
        'roughness': data['roughness'],        # (N,) Surface roughness
        'normalized_intensity': data['normalized_intensity'],
        'backscatter': data['backscatter'],
        
        # Segmentation
        'segmentation': data['segmentation'],  # (N,) Object segment IDs
        
        # Metadata
        'timestamp': data['timestamp'],
        'scan_id': data['scan_id'],
        'drone_position': data['drone_position'],
        'drone_velocity': data['drone_velocity'],
        
        # RGB-specific (may not exist in GPU LiDAR data)
        'rgb': data.get('rgb', None),          # (N, 3) RGB colors or None
        'specular': data.get('specular', None),  # (N,) Specular estimate or None
        'metallic': data.get('metallic', None),  # (N,) Metallic estimate or None
        'has_rgb': data.get('has_rgb', data.get('has_intensity', None)),  # Boolean mask
    }


def extract_features_basic(scan_data):
    """
    Extract basic feature set (works for both GPU and RGB data).
    
    Args:
        scan_data: Dictionary from load_scan()
    
    Returns:
        features: (N, 6) array [x, y, z, intensity, roughness, distance]
        labels: (N,) segmentation labels
    """
    features = np.column_stack([
        scan_data['points'],               # (N, 3) x, y, z
        scan_data['intensity'][:, None],   # (N, 1) intensity
        scan_data['roughness'][:, None],   # (N, 1) roughness
        scan_data['distance'][:, None],    # (N, 1) distance
    ])  # Result: (N, 6)
    
    labels = scan_data['segmentation']
    
    return features, labels


def extract_features_rgb_enhanced(scan_data):
    """
    Extract enhanced feature set with RGB data (only for RGB collector output).
    
    Args:
        scan_data: Dictionary from load_scan()
    
    Returns:
        features: (N, 12) array [x, y, z, intensity, roughness, distance, r, g, b, specular, metallic, has_rgb]
        labels: (N,) segmentation labels
    """
    if scan_data['rgb'] is None:
        raise ValueError("RGB data not available in this scan!")
    
    has_rgb = scan_data['has_rgb'].astype(np.float32)  # Convert bool to float
    
    features = np.column_stack([
        scan_data['points'],               # (N, 3) x, y, z
        scan_data['intensity'][:, None],   # (N, 1) intensity
        scan_data['roughness'][:, None],   # (N, 1) roughness
        scan_data['distance'][:, None],    # (N, 1) distance
        scan_data['rgb'].astype(np.float32) / 255.0,  # (N, 3) normalized RGB
        scan_data['specular'][:, None],    # (N, 1) specular
        scan_data['metallic'][:, None],    # (N, 1) metallic
        has_rgb[:, None],                  # (N, 1) has_rgb flag
    ])  # Result: (N, 12)
    
    labels = scan_data['segmentation']
    
    return features, labels


def filter_rgb_points_only(scan_data):
    """
    Filter to only points that have RGB data.
    
    Args:
        scan_data: Dictionary from load_scan()
    
    Returns:
        filtered_data: Dictionary with only RGB points
    """
    if scan_data['has_rgb'] is None:
        raise ValueError("No RGB mask available!")
    
    mask = scan_data['has_rgb']
    
    return {
        'points': scan_data['points'][mask],
        'intensity': scan_data['intensity'][mask],
        'roughness': scan_data['roughness'][mask],
        'distance': scan_data['distance'][mask],
        'segmentation': scan_data['segmentation'][mask],
        'rgb': scan_data['rgb'][mask] if scan_data['rgb'] is not None else None,
        'specular': scan_data['specular'][mask] if scan_data['specular'] is not None else None,
        'metallic': scan_data['metallic'][mask] if scan_data['metallic'] is not None else None,
    }


def load_dataset(data_dir, max_scans=None):
    """
    Load multiple scans from a directory.
    
    Args:
        data_dir: Path to directory with .npz files
        max_scans: Maximum number of scans to load (None = all)
    
    Returns:
        List of scan data dictionaries
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("scan_*.npz"))
    
    if max_scans is not None:
        npz_files = npz_files[:max_scans]
    
    print(f"Loading {len(npz_files)} scans from {data_dir}...")
    
    scans = []
    for npz_path in npz_files:
        scans.append(load_scan(npz_path))
    
    return scans


def combine_scans(scans, use_rgb=False):
    """
    Combine multiple scans into single feature matrix.
    
    Args:
        scans: List of scan data dictionaries
        use_rgb: If True, use RGB-enhanced features
    
    Returns:
        X: (total_points, feature_dim) features
        y: (total_points,) labels
    """
    all_features = []
    all_labels = []
    
    extract_func = extract_features_rgb_enhanced if use_rgb else extract_features_basic
    
    for scan in scans:
        features, labels = extract_func(scan)
        all_features.append(features)
        all_labels.append(labels)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_usage():
    """Example: Load and examine a single scan"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Load single scan
    scan = load_scan("lidar_data_rgb/scan_0000.npz")
    
    print(f"\nLoaded scan_0000.npz:")
    print(f"  Points:        {len(scan['points']):,}")
    print(f"  With RGB:      {np.sum(scan['has_rgb']):,}")
    print(f"  Segments:      {len(np.unique(scan['segmentation']))}")
    
    # Extract basic features
    X, y = extract_features_basic(scan)
    print(f"\nBasic features:")
    print(f"  Shape:         {X.shape}")
    print(f"  Features:      [x, y, z, intensity, roughness, distance]")
    print(f"  Labels shape:  {y.shape}")


def example_rgb_features():
    """Example: Use RGB-enhanced features"""
    print("\n" + "="*70)
    print("EXAMPLE 2: RGB-Enhanced Features")
    print("="*70)
    
    scan = load_scan("lidar_data_rgb/scan_0000.npz")
    
    # Extract RGB-enhanced features
    X, y = extract_features_rgb_enhanced(scan)
    print(f"\nRGB-enhanced features:")
    print(f"  Shape:         {X.shape}")
    print(f"  Features:      [x, y, z, intensity, roughness, distance,")
    print(f"                  r, g, b, specular, metallic, has_rgb]")
    print(f"  Labels shape:  {y.shape}")
    
    # Show RGB statistics
    has_rgb_mask = scan['has_rgb']
    print(f"\nRGB points only:")
    print(f"  Count:         {np.sum(has_rgb_mask):,}")
    print(f"  Mean RGB:      {scan['rgb'][has_rgb_mask].mean(axis=0)}")
    print(f"  Mean specular: {scan['specular'][has_rgb_mask].mean():.3f}")


def example_filter_rgb_only():
    """Example: Work with RGB points only"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Filter RGB Points Only")
    print("="*70)
    
    scan = load_scan("lidar_data_rgb/scan_0000.npz")
    
    # Filter to RGB points
    rgb_only = filter_rgb_points_only(scan)
    
    print(f"\nOriginal scan:   {len(scan['points']):,} points")
    print(f"RGB-only scan:   {len(rgb_only['points']):,} points")
    
    # Now all points have valid RGB data
    print(f"\nRGB-only statistics:")
    print(f"  Mean RGB:      {rgb_only['rgb'].mean(axis=0)}")
    print(f"  Mean intensity:{rgb_only['intensity'].mean():.1f}")
    print(f"  Mean roughness:{rgb_only['roughness'].mean():.3f}")


def example_load_full_dataset():
    """Example: Load and combine entire dataset"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Load Full Dataset")
    print("="*70)
    
    # Load first 10 scans
    scans = load_dataset("lidar_data_rgb", max_scans=10)
    
    print(f"\nLoaded {len(scans)} scans")
    
    # Combine into single dataset
    X_basic, y_basic = combine_scans(scans, use_rgb=False)
    X_rgb, y_rgb = combine_scans(scans, use_rgb=True)
    
    print(f"\nCombined dataset:")
    print(f"  Basic features:   {X_basic.shape}")
    print(f"  RGB features:     {X_rgb.shape}")
    print(f"  Labels:           {y_basic.shape}")
    
    # Class distribution
    unique, counts = np.unique(y_basic, return_counts=True)
    print(f"\nClass distribution:")
    for seg_id, count in zip(unique, counts):
        print(f"  Segment {seg_id:3d}:    {count:,} points ({100*count/len(y_basic):.1f}%)")


def example_train_test_split():
    """Example: Prepare data for ML training"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Train/Test Split")
    print("="*70)
    
    # Load dataset
    scans = load_dataset("lidar_data_rgb", max_scans=20)
    
    # Split 80/20
    split_idx = int(0.8 * len(scans))
    train_scans = scans[:split_idx]
    test_scans = scans[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Total scans:   {len(scans)}")
    print(f"  Train scans:   {len(train_scans)}")
    print(f"  Test scans:    {len(test_scans)}")
    
    # Combine train and test
    X_train, y_train = combine_scans(train_scans, use_rgb=True)
    X_test, y_test = combine_scans(test_scans, use_rgb=True)
    
    print(f"\nTrain set:")
    print(f"  Features:      {X_train.shape}")
    print(f"  Labels:        {y_train.shape}")
    
    print(f"\nTest set:")
    print(f"  Features:      {X_test.shape}")
    print(f"  Labels:        {y_test.shape}")
    
    print(f"\nReady for training!")
    print("  Example: model.fit(X_train, y_train)")
    print("           predictions = model.predict(X_test)")


def main():
    """Run all examples"""
    
    data_dir = Path("lidar_data_rgb")
    
    if not data_dir.exists() or not list(data_dir.glob("scan_*.npz")):
        print("❌ No RGB LiDAR data found!")
        print("   Run rgb_lidar_collector.py first to generate data.")
        return
    
    print("\n" + "="*70)
    print("RGB LIDAR DATA LOADING EXAMPLES")
    print("="*70)
    
    try:
        example_basic_usage()
        example_rgb_features()
        example_filter_rgb_only()
        example_load_full_dataset()
        example_train_test_split()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETE!")
        print("="*70)
        print("""
Summary:
  ✓ Loaded RGB LiDAR data from .npz files
  ✓ Extracted basic features (6D)
  ✓ Extracted RGB-enhanced features (12D)
  ✓ Filtered to RGB points only
  ✓ Combined multiple scans
  ✓ Created train/test split

Next steps:
  1. Adapt these examples to your AI pipeline
  2. Try different feature combinations
  3. Use segmentation labels for supervised learning
  4. Leverage rgb, specular, metallic for better classification
""")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
