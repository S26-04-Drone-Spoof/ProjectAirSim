"""
RGB + Depth Fusion for Material-Based Object Detection

This script fuses CPU LiDAR point clouds with RGB camera data to extract
material properties based on color and texture. Replaces the failed GPU LiDAR
intensity approach with a simpler, proven computer vision technique.

Workflow:
1. Get CPU LiDAR point cloud (accurate 3D geometry)
2. Get DownCamera RGB image (material appearance)
3. Project 3D points onto 2D camera plane
4. Extract RGB values at projected pixels
5. Classify materials from color/texture patterns
"""

import asyncio
import numpy as np
import time
import cv2
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.types import ImageType


class RGBDepthFusion:
    """Fusion of CPU LiDAR geometry with RGB camera appearance"""
    
    def __init__(self, drone, camera_name="DownCamera"):
        self.drone = drone
        self.camera_name = camera_name
        
        # DownCamera configuration from robot config
        self.image_width = 400
        self.image_height = 225
        self.fov_deg = 90
        
        # Camera intrinsics
        self.fov_rad = np.deg2rad(self.fov_deg)
        self.focal_length = (self.image_width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        # Camera extrinsics: DownCamera points straight down (0, -90, 0) RPY
        # Camera frame: X=right, Y=down (into image), Z=forward (viewing direction)
        # Drone frame: X=forward, Y=right, Z=down
        # Rotation: 90° pitch down means drone X → camera Z, drone Z → camera Y
        self.camera_to_drone_rot = self._rpy_to_rotation_matrix(0, -90, 0)
        self.drone_to_camera_rot = self.camera_to_drone_rot.T
        
        # Storage for latest LiDAR scan
        self.latest_lidar_points = None
        
        print(f"RGB+Depth Fusion initialized")
        print(f"  Camera: {camera_name}")
        print(f"  Image size: {self.image_width}x{self.image_height}")
        print(f"  FOV: {self.fov_deg}°")
        print(f"  Focal length: {self.focal_length:.2f} pixels")
    
    def _rpy_to_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg):
        """Convert roll-pitch-yaw (degrees) to rotation matrix"""
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        # Roll (rotation around X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (rotation around Y)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (rotation around Z)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def lidar_callback(self, topic, lidar_data):
        """Callback to capture latest CPU LiDAR scan"""
        if lidar_data is None or "point_cloud" not in lidar_data:
            return
        
        pts = np.array(lidar_data["point_cloud"], dtype=np.float32).reshape(-1, 3)
        if len(pts) > 0:
            self.latest_lidar_points = pts
    
    def get_cpu_lidar_pointcloud(self):
        """Get latest CPU LiDAR point cloud"""
        return self.latest_lidar_points
    
    def get_rgb_image(self):
        """Get RGB image from DownCamera"""
        images = self.drone.get_images(self.camera_name, [ImageType.SCENE])
        
        if ImageType.SCENE not in images:
            return None
        
        img_rgb = unpack_image(images[ImageType.SCENE])
        return img_rgb
    
    def project_points_to_camera(self, points_drone_frame):
        """
        Project 3D points (in drone frame) onto 2D camera plane
        
        Args:
            points_drone_frame: Nx3 array of 3D points in drone coordinate frame
            
        Returns:
            pixel_coords: Nx2 array of (u, v) pixel coordinates
            valid_mask: Boolean mask indicating which points are visible
        """
        # Transform points from drone frame to camera frame
        points_camera_frame = (self.drone_to_camera_rot @ points_drone_frame.T).T
        
        # Camera frame: X=right, Y=down, Z=forward
        # Points with Z > 0 are in front of camera
        valid_depth = points_camera_frame[:, 2] > 0.1  # At least 10cm in front
        
        # Perspective projection: (X, Y, Z) -> (u, v)
        # u = fx * (X/Z) + cx
        # v = fy * (Y/Z) + cy
        # (For square pixels, fx = fy = focal_length)
        
        u = self.focal_length * (points_camera_frame[:, 0] / points_camera_frame[:, 2]) + self.cx
        v = self.focal_length * (points_camera_frame[:, 1] / points_camera_frame[:, 2]) + self.cy
        
        # Check if pixels are within image bounds
        valid_u = (u >= 0) & (u < self.image_width)
        valid_v = (v >= 0) & (v < self.image_height)
        
        valid_mask = valid_depth & valid_u & valid_v
        
        pixel_coords = np.column_stack([u, v])
        
        return pixel_coords, valid_mask
    
    def extract_rgb_at_points(self, rgb_image, pixel_coords, valid_mask):
        """
        Extract RGB values at projected pixel locations
        
        Args:
            rgb_image: HxWx3 RGB image
            pixel_coords: Nx2 array of (u, v) pixel coordinates
            valid_mask: Boolean mask for valid projections
            
        Returns:
            rgb_values: Nx3 array of RGB values (0-255)
        """
        rgb_values = np.zeros((len(pixel_coords), 3), dtype=np.uint8)
        
        if not np.any(valid_mask):
            return rgb_values
        
        # Get valid pixel coordinates (round to nearest integer)
        valid_pixels = pixel_coords[valid_mask].astype(int)
        u = valid_pixels[:, 0]
        v = valid_pixels[:, 1]
        
        # Clamp to image bounds (safety)
        u = np.clip(u, 0, self.image_width - 1)
        v = np.clip(v, 0, self.image_height - 1)
        
        # Extract RGB values (note: OpenCV uses BGR, but unpack_image gives RGB)
        rgb_values[valid_mask] = rgb_image[v, u]
        
        return rgb_values
    
    def classify_material_from_rgb(self, rgb_values, valid_mask):
        """
        Classify material type from RGB values
        
        Simple heuristics (can be replaced with ML classifier later):
        - Rough surfaces: Medium brightness, varied colors
        - Smooth surfaces: High/low brightness, uniform colors
        - Metallic: High brightness + color saturation
        - Concrete: Gray (low saturation)
        - Vegetation: Green dominant
        - Asphalt: Dark gray/black
        
        Returns:
            material_scores: Dictionary with material probabilities per point
        """
        n_points = len(rgb_values)
        
        # Convert to HSV for easier material discrimination
        rgb_for_cv2 = rgb_values.reshape(1, -1, 3)
        hsv = cv2.cvtColor(rgb_for_cv2, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        h = hsv[:, 0]  # Hue [0, 179]
        s = hsv[:, 1]  # Saturation [0, 255]
        v = hsv[:, 2]  # Value/Brightness [0, 255]
        
        # Material classification scores
        materials = {
            'metallic': np.zeros(n_points, dtype=np.float32),
            'rough': np.zeros(n_points, dtype=np.float32),
            'smooth': np.zeros(n_points, dtype=np.float32),
            'concrete': np.zeros(n_points, dtype=np.float32),
            'vegetation': np.zeros(n_points, dtype=np.float32),
            'asphalt': np.zeros(n_points, dtype=np.float32)
        }
        
        if not np.any(valid_mask):
            return materials
        
        # Metallic: High brightness + some saturation
        materials['metallic'][valid_mask] = (
            (v > 150).astype(float) * 
            (s > 30).astype(float) * 
            (s < 150).astype(float)
        )[valid_mask]
        
        # Smooth: Very high or very low brightness, low saturation variation
        materials['smooth'][valid_mask] = (
            ((v > 180) | (v < 50)).astype(float) * 
            (s < 80).astype(float)
        )[valid_mask]
        
        # Rough: Medium brightness, varied colors
        materials['rough'][valid_mask] = (
            (v > 80).astype(float) * 
            (v < 180).astype(float) * 
            (s > 50).astype(float)
        )[valid_mask]
        
        # Concrete: Gray (low saturation, medium brightness)
        materials['concrete'][valid_mask] = (
            (s < 40).astype(float) * 
            (v > 60).astype(float) * 
            (v < 180).astype(float)
        )[valid_mask]
        
        # Vegetation: Green hue
        materials['vegetation'][valid_mask] = (
            ((h > 35) & (h < 85)).astype(float) * 
            (s > 40).astype(float)
        )[valid_mask]
        
        # Asphalt: Dark, low saturation
        materials['asphalt'][valid_mask] = (
            (v < 80).astype(float) * 
            (s < 40).astype(float)
        )[valid_mask]
        
        return materials
    
    def fuse_lidar_rgb(self, save_colored_ply=None):
        """
        Main fusion function: combine CPU LiDAR geometry with RGB appearance
        
        Args:
            save_colored_ply: Optional filename to save colored point cloud
            
        Returns:
            Dictionary with:
                - points: Nx3 LiDAR points
                - rgb: Nx3 RGB values
                - valid_mask: Boolean mask for points with valid RGB
                - materials: Material classification scores
        """
        # Get CPU LiDAR point cloud
        points = self.get_cpu_lidar_pointcloud()
        if points is None or len(points) == 0:
            print("ERROR: No LiDAR points received")
            return None
        
        # Get RGB image
        rgb_image = self.get_rgb_image()
        if rgb_image is None:
            print("ERROR: No RGB image received")
            return None
        
        # Project 3D points onto 2D camera plane
        pixel_coords, valid_mask = self.project_points_to_camera(points)
        
        # Extract RGB values at projected locations
        rgb_values = self.extract_rgb_at_points(rgb_image, pixel_coords, valid_mask)
        
        # Classify materials from RGB
        materials = self.classify_material_from_rgb(rgb_values, valid_mask)
        
        # Stats
        n_total = len(points)
        n_valid = np.sum(valid_mask)
        pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0
        
        print(f"\nFusion results:")
        print(f"  Total LiDAR points: {n_total}")
        print(f"  Points with valid RGB: {n_valid} ({pct_valid:.1f}%)")
        print(f"  RGB image size: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
        
        # Material distribution
        print(f"\nMaterial classification:")
        for mat_name, scores in materials.items():
            n_classified = np.sum(scores > 0.5)
            pct_classified = 100.0 * n_classified / n_valid if n_valid > 0 else 0
            print(f"  {mat_name:12s}: {n_classified:6d} points ({pct_classified:5.1f}%)")
        
        # Save colored point cloud if requested
        if save_colored_ply:
            self._save_colored_ply(points, rgb_values, valid_mask, save_colored_ply)
            print(f"\nSaved colored point cloud: {save_colored_ply}")
        
        return {
            'points': points,
            'rgb': rgb_values,
            'valid_mask': valid_mask,
            'materials': materials,
            'rgb_image': rgb_image,
            'pixel_coords': pixel_coords
        }
    
    def _save_colored_ply(self, points, rgb, valid_mask, filename):
        """Save point cloud with RGB colors in PLY format"""
        with open(filename, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {np.sum(valid_mask)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write points with RGB
            valid_points = points[valid_mask]
            valid_rgb = rgb[valid_mask]
            
            for i in range(len(valid_points)):
                x, y, z = valid_points[i]
                r, g, b = valid_rgb[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")


async def main():
    """Test RGB+Depth fusion with live data"""
    
    print("="*60)
    print("RGB + Depth Fusion for Material Detection")
    print("="*60)
    
    client = ProjectAirSimClient()
    client.connect()
    
    world = World(client, "scene_lidar_drone.jsonc", delay_after_load_sec=2)
    drone = Drone(client, world, "Drone1")
    
    # Create fusion object
    fusion = RGBDepthFusion(drone)
    
    # Subscribe to CPU LiDAR
    client.subscribe(
        drone.sensors["cpu_lidar1"]["lidar"],
        fusion.lidar_callback
    )
    
    # Takeoff sequence
    print("\nTaking off...")
    drone.enable_api_control()
    drone.arm()
    
    takeoff_task = await drone.takeoff_async()
    await takeoff_task
    
    print("Hovering at altitude...")
    cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    move_task = await drone.move_to_position_async(
        north=cur_pos["x"], east=cur_pos["y"], down=-10.0, velocity=2.0
    )
    await move_task
    
    await asyncio.sleep(2.0)  # Stabilize and let LiDAR scan
    
    print("\nStarting RGB+Depth fusion...")
    
    # Capture and fuse multiple scans
    for i in range(5):
        print(f"\n--- Scan {i+1}/5 ---")
        
        # Wait a bit for LiDAR to accumulate
        await asyncio.sleep(0.5)
        
        result = fusion.fuse_lidar_rgb(save_colored_ply=f"colored_pointcloud_{i+1}.ply")
        
        if result:
            # Show RGB image with projected points overlay
            rgb_img = result['rgb_image'].copy()
            pixel_coords = result['pixel_coords']
            valid = result['valid_mask']
            
            # Draw valid projections on image
            valid_pixels = pixel_coords[valid].astype(int)
            
            for u, v in valid_pixels:
                if 0 <= u < rgb_img.shape[1] and 0 <= v < rgb_img.shape[0]:
                    cv2.circle(rgb_img, (u, v), 1, (0, 255, 0), -1)
            
            # Save visualization (OpenCV expects BGR)
            cv2.imwrite(f"rgb_projection_{i+1}.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            print(f"  Saved: rgb_projection_{i+1}.png")
        
        await asyncio.sleep(1.0)  # Wait between scans
    
    print("\n" + "="*60)
    print("RGB+Depth fusion complete!")
    print("Check .ply files in CloudCompare or MeshLab")
    print("="*60)
    
    # Unsubscribe from LiDAR before landing
    print("\nUnsubscribing from sensors...")
    client.unsubscribe(drone.sensors["cpu_lidar1"]["lidar"])
    
    # Land
    print("Landing...")
    land_task = await drone.land_async()
    await land_task
    
    drone.disarm()
    drone.disable_api_control()
    
    print("Disconnecting...")
    client.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
