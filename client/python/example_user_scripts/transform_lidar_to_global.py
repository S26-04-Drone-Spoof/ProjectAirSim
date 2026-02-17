"""
Example: Transform LiDAR scans to global coordinate frame using ground truth pose.

This script demonstrates how to:
1. Load LiDAR scan data with ground truth pose
2. Transform sensor-frame points to global frame
3. Build a unified point cloud map from multiple scans
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import open3d as o3d


def load_scan(scan_file):
    """Load a single LiDAR scan with metadata"""
    data = np.load(scan_file)
    return {
        'points': data['points'],  # (N, 3) in sensor frame
        'intensity': data['intensity'],
        'segmentation': data['segmentation'],
        'timestamp': data['timestamp'],
        'drone_position': data['drone_position'],  # (3,) - x, y, z
        'drone_orientation_quat': data['drone_orientation_quat'],  # (4,) - w, x, y, z
        'drone_orientation_euler': data['drone_orientation_euler'],  # (3,) - roll, pitch, yaw
    }


def transform_to_global(points_sensor, drone_position, drone_orientation_quat):
    """
    Transform points from sensor frame to global frame.

    Args:
        points_sensor: (N, 3) array of points in sensor coordinate frame
        drone_position: (3,) array - drone XYZ position in global frame
        drone_orientation_quat: (4,) array - quaternion (w, x, y, z)

    Returns:
        points_global: (N, 3) array of points in global coordinate frame
    """
    # Create rotation object from quaternion (w, x, y, z format)
    rotation = Rotation.from_quat([
        drone_orientation_quat[1],  # x
        drone_orientation_quat[2],  # y
        drone_orientation_quat[3],  # z
        drone_orientation_quat[0],  # w
    ])  # scipy uses (x, y, z, w) format

    # Get rotation matrix
    rotation_matrix = rotation.as_matrix()

    # Transform: global = position + R @ sensor
    points_global = drone_position + (rotation_matrix @ points_sensor.T).T

    return points_global


def build_global_map(dataset_dir, max_scans=None):
    """
    Build a unified global point cloud map from multiple scans.

    Args:
        dataset_dir: Path to directory containing scan_*.npz files
        max_scans: Maximum number of scans to process (None = all)

    Returns:
        global_points: (N, 3) all points in global frame
        global_intensities: (N,) corresponding intensities
        global_segmentation: (N,) corresponding segmentation IDs
    """
    dataset_path = Path(dataset_dir)
    scan_files = sorted(dataset_path.glob("scan_*.npz"))

    if max_scans:
        scan_files = scan_files[:max_scans]

    print(f"Processing {len(scan_files)} scans from {dataset_dir}")

    all_global_points = []
    all_intensities = []
    all_segmentation = []

    for scan_file in scan_files:
        # Load scan
        scan = load_scan(scan_file)

        # Transform to global frame
        global_points = transform_to_global(
            scan['points'],
            scan['drone_position'],
            scan['drone_orientation_quat']
        )

        # Accumulate
        all_global_points.append(global_points)
        all_intensities.append(scan['intensity'])
        all_segmentation.append(scan['segmentation'])

        print(f"  {scan_file.name}: {len(scan['points'])} points at position {scan['drone_position']}")

    # Concatenate all scans
    global_points = np.vstack(all_global_points)
    global_intensities = np.concatenate(all_intensities)
    global_segmentation = np.concatenate(all_segmentation)

    print(f"\nGlobal map built: {len(global_points):,} total points")

    return global_points, global_intensities, global_segmentation


def visualize_global_map(global_points, global_intensities):
    """Visualize the global point cloud map using Open3D"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_points)

    # Color by intensity (normalized)
    intensity_normalized = (global_intensities - global_intensities.min()) / (
        global_intensities.max() - global_intensities.min() + 1e-6
    )
    colors = np.stack([intensity_normalized] * 3, axis=1)  # Grayscale
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd],
                                     window_name="Global Point Cloud Map",
                                     width=1280, height=720)


def save_global_map(global_points, global_intensities, global_segmentation, output_file):
    """Save the global map to a file"""
    np.savez_compressed(
        output_file,
        points=global_points,
        intensity=global_intensities,
        segmentation=global_segmentation
    )
    print(f"Global map saved to {output_file}")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Process a scenario from the dataset
    DATASET_DIR = "lidar_dataset_unspoofed/scenario_0000_forward_scan"
    OUTPUT_FILE = "global_map.npz"

    # Build global map from all scans
    global_points, global_intensities, global_segmentation = build_global_map(
        DATASET_DIR,
        max_scans=None  # Process all scans
    )

    # Save the global map
    save_global_map(global_points, global_intensities, global_segmentation, OUTPUT_FILE)

    # Visualize (optional - requires display)
    print("\nLaunching 3D visualization...")
    visualize_global_map(global_points, global_intensities)

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Load the global map for your ML model")
    print("2. Run spoof detection on the unified point cloud")
    print("3. For SLAM comparison: run ICP-based alignment and compare with ground truth")
