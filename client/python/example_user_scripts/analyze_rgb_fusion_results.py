"""
Analyze RGB+Depth Fusion Results
Shows statistics and creates visualizations of the output
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_ply_file(filename):
    """Analyze a PLY point cloud file"""
    if not Path(filename).exists():
        print(f"❌ {filename} not found")
        return None
    
    # Read PLY file
    points = []
    colors = []
    reading_data = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                reading_data = True
                continue
            if reading_data and line:
                parts = line.split()
                if len(parts) == 6:
                    x, y, z, r, g, b = map(float, parts)
                    points.append([x, y, z])
                    colors.append([r, g, b])
    
    if len(points) == 0:
        print(f"⚠️  {filename}: EMPTY (0 points)")
        return None
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"\n✓ {filename}:")
    print(f"  Points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Color statistics
    gray = colors.mean(axis=1)
    print(f"  Brightness: mean={gray.mean():.1f}, std={gray.std():.1f}")
    
    # Count material types based on color
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    gray_mask = (np.abs(r - g) < 10) & (np.abs(g - b) < 10)
    n_gray = np.sum(gray_mask)
    print(f"  Gray points (concrete/asphalt): {n_gray} ({100*n_gray/len(points):.1f}%)")
    
    return {
        'filename': filename,
        'points': points,
        'colors': colors,
        'n_points': len(points)
    }


def analyze_projection_image(filename):
    """Analyze a RGB projection visualization"""
    if not Path(filename).exists():
        print(f"❌ {filename} not found")
        return None
    
    img = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Count green pixels (projected points)
    green_mask = (img_rgb[:, :, 1] > 200) & (img_rgb[:, :, 0] < 100) & (img_rgb[:, :, 2] < 100)
    n_green_pixels = np.sum(green_mask)
    
    print(f"\n✓ {filename}:")
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"  Green pixels (projections): {n_green_pixels}")
    print(f"  Coverage: {100*n_green_pixels/(img.shape[0]*img.shape[1]):.2f}%")
    
    return {
        'filename': filename,
        'image': img_rgb,
        'n_projections': n_green_pixels
    }


def create_summary_visualization(ply_results, img_results):
    """Create a summary visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    fig.suptitle("RGB+Depth Fusion Results Summary", fontsize=16, fontweight='bold')
    
    # Plot projection images
    for i, result in enumerate(img_results[:3]):
        if result and i < 3:
            ax = axes[0, i]
            ax.imshow(result['image'])
            ax.set_title(f"Scan {i+1}\n{result['n_projections']} projections")
            ax.axis('off')
    
    # Plot point cloud top views
    for i, result in enumerate(ply_results[:3]):
        if result and i < 3:
            ax = axes[1, i]
            points = result['points']
            colors = result['colors'] / 255.0  # Normalize to [0, 1]
            
            # Top view (X-Y plane)
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.5)
            ax.set_xlabel('X (North)')
            ax.set_ylabel('Y (East)')
            ax.set_title(f"{result['n_points']} points")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fusion_summary.png', dpi=150)
    print(f"\n✓ Saved summary visualization: fusion_summary.png")
    plt.close()


def main():
    print("="*60)
    print("RGB+Depth Fusion Results Analysis")
    print("="*60)
    
    # Analyze all PLY files
    print("\n📊 Analyzing Point Clouds:")
    ply_results = []
    for i in range(1, 6):
        result = analyze_ply_file(f"colored_pointcloud_{i}.ply")
        if result:
            ply_results.append(result)
    
    # Analyze all projection images
    print("\n📸 Analyzing Projection Images:")
    img_results = []
    for i in range(1, 6):
        result = analyze_projection_image(f"rgb_projection_{i}.png")
        if result:
            img_results.append(result)
    
    # Overall statistics
    print("\n" + "="*60)
    print("Overall Statistics:")
    total_points = sum(r['n_points'] for r in ply_results)
    successful_scans = len(ply_results)
    failed_scans = 5 - successful_scans
    
    print(f"  Successful scans: {successful_scans}/5")
    print(f"  Failed scans: {failed_scans}/5")
    print(f"  Total points captured: {total_points}")
    if successful_scans > 0:
        print(f"  Average points per scan: {total_points/successful_scans:.1f}")
    
    # Create visualization
    if ply_results and img_results:
        print("\n📈 Creating summary visualization...")
        create_summary_visualization(ply_results, img_results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("\nKey findings:")
    if failed_scans > 0:
        print(f"  ⚠️  {failed_scans} scans had no valid points")
        print("     → CPU LiDAR may not have accumulated data yet")
        print("     → Try increasing wait time between scans")
    if successful_scans > 0:
        avg_pts = total_points / successful_scans
        if avg_pts < 500:
            print(f"  ⚠️  Low point density ({avg_pts:.0f} pts/scan)")
            print("     → DownCamera FOV may not overlap well with LiDAR")
            print("     → Most LiDAR points may be outside camera view")
        else:
            print(f"  ✓ Good point density ({avg_pts:.0f} pts/scan)")
    print("="*60)


if __name__ == "__main__":
    main()
