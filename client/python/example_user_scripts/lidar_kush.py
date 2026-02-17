"""
LiDAR Data Collection Script - Enhanced Version
Flies drone in a pattern while collecting and storing LiDAR point cloud data.
Stores raw data + derived material features for ML dataset generation.

GROUND TRUTH POSE DATA:
Each scan includes the drone's complete pose at scan time:
- drone_position: (x, y, z) in meters (NED frame)
- drone_orientation_quat: (w, x, y, z) quaternion
- drone_orientation_euler: (roll, pitch, yaw) in radians

COORDINATE TRANSFORMATION:
To transform LiDAR points from sensor frame to global frame:
1. Create rotation matrix from quaternion or Euler angles
2. Apply: global_point = drone_position + rotation_matrix @ sensor_point
3. This allows building a unified map from all scans

See: scipy.spatial.transform.Rotation for easy rotation matrix conversion
"""

import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import shutil
from scipy.spatial import cKDTree
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, quaternion_to_rpy
from projectairsim.image_utils import ImageDisplay
from projectairsim.lidar_utils import LidarDisplay


class LidarDataCollector:
    """
    Collects LiDAR data and stores in enhanced format for ML training.

    Stores:
    - Raw sensor data (points, intensity, segmentation)
    - Derived features (normalized_intensity, roughness, backscatter, etc.)
    - Metadata (scene, drone state including position and orientation)

    Features are computed but NOT normalized - normalization is left to ML engineer.
    """

    def __init__(self, output_dir="lidar_data", scene_name="scene_lidar_drone",
                 compute_features=True, clear_existing=True):
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.compute_features = compute_features

        # Clear existing data if requested (for single runs)
        # Set clear_existing=False for test bench to preserve previous scenarios
        if clear_existing and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            projectairsim_log().info(f"Cleared existing data from {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scan_count = 0
        self.collected_scans = []
        self.start_time = time.time()
        self.drone = None  # Will be set externally

        projectairsim_log().info(f"LiDAR data will be saved to: {self.output_dir.absolute()}")
        projectairsim_log().info(f"Feature computation: {'ENABLED' if compute_features else 'DISABLED'}")

    def set_drone(self, drone):
        """Set drone reference to access state information"""
        self.drone = drone

    def _compute_distance(self, points):
        """Compute Euclidean distance from sensor origin"""
        distance = np.linalg.norm(points, axis=1)
        distance = np.maximum(distance, 0.1)  # Avoid division by zero
        return distance

    def _compute_normalized_intensity(self, intensity, distance):
        """
        Compute distance-normalized intensity (relative reflectivity).
        Removes distance effect: I_normalized = I_raw * distance²
        """
        return intensity * (distance ** 2)

    def _compute_surface_roughness(self, points, k=10):
        """
        Compute surface roughness using local PCA.
        Returns roughness in [0, 1] where:
        - 0 = perfectly planar surface
        - 1 = highly scattered/rough surface
        """
        if len(points) < k:
            return np.zeros(len(points), dtype=np.float32)

        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(k, len(points)))

        roughness = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues = np.linalg.eigvalsh(cov)
            # Roughness = ratio of smallest to largest eigenvalue
            roughness[i] = eigenvalues[0] / (eigenvalues[2] + 1e-6)

        return roughness

    def _compute_backscatter_coefficient(self, normalized_intensity, roughness):
        """
        Compute backscatter coefficient (material signature).
        Accounts for both reflectivity and surface texture.
        """
        return normalized_intensity / (1.0 + roughness)

    def _compute_intensity_variance(self, points, intensity, k=10):
        """
        Compute local intensity variance.
        Indicates material homogeneity.
        """
        if len(points) < k:
            return np.zeros(len(points), dtype=np.float32)

        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(k, len(points)))

        intensity_var = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            neighbor_intensities = intensity[indices[i]]
            intensity_var[i] = np.var(neighbor_intensities)

        return intensity_var

    def _estimate_surface_normals(self, points, k=10):
        """Estimate surface normals using local PCA"""
        if len(points) < k:
            return np.zeros_like(points, dtype=np.float32)

        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(k, len(points)))

        normals = np.zeros_like(points, dtype=np.float32)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            _, eigenvectors = np.linalg.eigh(cov)
            # Normal is the eigenvector with smallest eigenvalue
            normals[i] = eigenvectors[:, 0]

        return normals

    def _compute_viewing_angles(self, points, normals):
        """
        Compute angle between surface normal and viewing direction.
        Important for understanding Lambert's Law compliance.
        """
        # Viewing direction = from sensor (0,0,0) to point
        view_dirs = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-6)

        # Angle between normal and viewing direction
        cos_angles = np.abs(np.sum(normals * view_dirs, axis=1))
        angles = np.arccos(np.clip(cos_angles, 0, 1))

        return angles

    def process_and_store(self, topic, lidar_data):
        """Process LiDAR data with full feature extraction and store"""
        if lidar_data is None:
            return

        start_process = time.time()

        # Generate timestamp relative to start
        timestamp = time.time() - self.start_time

        # === RAW DATA EXTRACTION ===
        points = np.array(lidar_data["point_cloud"], dtype=np.float32)
        points_nx3 = points.reshape(-1, 3)
        intensity = np.array(lidar_data["intensity_cloud"], dtype=np.float32)
        segmentation = np.array(lidar_data["segmentation_cloud"], dtype=np.int32)

        num_points = points_nx3.shape[0]

        # Get drone state if available
        drone_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        drone_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        drone_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
        drone_orientation_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw

        if self.drone is not None:
            try:
                state = self.drone.get_ground_truth_kinematics()

                # Position
                drone_position = np.array([
                    state["pose"]["position"]["x"],
                    state["pose"]["position"]["y"],
                    state["pose"]["position"]["z"],
                ], dtype=np.float32)

                # Velocity
                drone_velocity = np.array([
                    state["twist"]["linear"]["x"],
                    state["twist"]["linear"]["y"],
                    state["twist"]["linear"]["z"],
                ], dtype=np.float32)

                # Orientation (quaternion)
                q = state["pose"]["orientation"]
                drone_orientation_quat = np.array([
                    q["w"], q["x"], q["y"], q["z"]
                ], dtype=np.float32)

                # Convert quaternion to Euler angles (roll, pitch, yaw) using built-in utility
                roll, pitch, yaw = quaternion_to_rpy(q["w"], q["x"], q["y"], q["z"])
                drone_orientation_euler = np.array([roll, pitch, yaw], dtype=np.float32)

            except Exception as e:
                projectairsim_log().warning(f"Could not get drone state: {e}")
                pass  # Use defaults if state unavailable

        # === DERIVED FEATURES ===
        if self.compute_features:
            projectairsim_log().info(f"Scan {self.scan_count}: Computing features for {num_points} points...")

            # 1. Distance
            distance = self._compute_distance(points_nx3)

            # 2. Normalized Intensity (Reflectivity)
            normalized_intensity = self._compute_normalized_intensity(intensity, distance)

            # 3. Surface Roughness
            roughness = self._compute_surface_roughness(points_nx3, k=10)

            # 4. Backscatter Coefficient
            backscatter = self._compute_backscatter_coefficient(normalized_intensity, roughness)

            # 5. Intensity Variance
            intensity_variance = self._compute_intensity_variance(points_nx3, intensity, k=10)

            # 6. Surface Normals and Viewing Angles
            normals = self._estimate_surface_normals(points_nx3, k=10)
            viewing_angle = self._compute_viewing_angles(points_nx3, normals)
        else:
            # Minimal features
            distance = self._compute_distance(points_nx3)
            normalized_intensity = self._compute_normalized_intensity(intensity, distance)
            roughness = np.zeros(num_points, dtype=np.float32)
            backscatter = np.zeros(num_points, dtype=np.float32)
            intensity_variance = np.zeros(num_points, dtype=np.float32)
            viewing_angle = np.zeros(num_points, dtype=np.float32)

        # === SAVE TO DISK ===
        filename = self.output_dir / f"scan_{self.scan_count:04d}.npz"
        np.savez_compressed(
            filename,
            # === RAW DATA ===
            points=points_nx3,
            intensity=intensity,
            segmentation=segmentation,
            timestamp=timestamp,
            scan_id=self.scan_count,

            # === DERIVED FEATURES (not normalized) ===
            distance=distance,
            normalized_intensity=normalized_intensity,
            roughness=roughness,
            backscatter=backscatter,
            intensity_variance=intensity_variance,
            viewing_angle=viewing_angle,

            # === METADATA ===
            scene_name=self.scene_name,
            drone_position=drone_position,
            drone_velocity=drone_velocity,
            drone_orientation_quat=drone_orientation_quat,  # quaternion (w, x, y, z)
            drone_orientation_euler=drone_orientation_euler,  # Euler angles (roll, pitch, yaw) in radians
        )

        process_time = time.time() - start_process

        # Store summary for later
        self.collected_scans.append({
            'scan_id': self.scan_count,
            'timestamp': timestamp,
            'num_points': num_points,
            'process_time': process_time
        })

        projectairsim_log().info(
            f"Scan {self.scan_count}: {num_points} points saved to {filename.name} "
            f"(processed in {process_time:.2f}s)"
        )

        self.scan_count += 1
    
    def save_summary(self):
        """Save summary of all collected scans"""
        if not self.collected_scans:
            projectairsim_log().warning("No scans collected!")
            return

        # Create summary
        total_points = sum(scan['num_points'] for scan in self.collected_scans)
        total_process_time = sum(scan['process_time'] for scan in self.collected_scans)
        summary_file = self.output_dir / "collection_summary.txt"

        with open(summary_file, 'w') as f:
            f.write(f"LiDAR Data Collection Summary\n")
            f.write(f"Collection Date: {datetime.now()}\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Scene: {self.scene_name}\n")
            f.write(f"Total Scans: {self.scan_count}\n")
            f.write(f"Total Points: {total_points:,}\n")
            f.write(f"Average Points/Scan: {total_points / self.scan_count:.1f}\n")
            f.write(f"Total Processing Time: {total_process_time:.2f}s\n")
            f.write(f"Average Processing Time/Scan: {total_process_time / self.scan_count:.2f}s\n\n")
            f.write(f"Individual Scan Details:\n")
            f.write("-" * 60 + "\n")
            for scan in self.collected_scans:
                f.write(f"  Scan {scan['scan_id']:04d}: "
                       f"{scan['num_points']:6d} points at t={scan['timestamp']:.3f}s "
                       f"(processed in {scan['process_time']:.2f}s)\n")

        projectairsim_log().info(f"Summary saved to {summary_file}")
        projectairsim_log().info(f"Collected {self.scan_count} scans with {total_points:,} total points")


async def main():
    # Configuration
    ENABLE_VISUALIZATION = False  # Set to True to see camera/LiDAR windows

    # Create a Project AirSim client
    client = ProjectAirSimClient()

    # Initialize displays only if visualization enabled
    image_display = None
    lidar_display = None

    if ENABLE_VISUALIZATION:
        image_display = ImageDisplay()
        lidar_subwin = image_display.get_subwin_info(2)
        lidar_display = LidarDisplay(
            x=lidar_subwin["x"], y=lidar_subwin["y"] + 30
        )

    # Initialize data collector
    collector = LidarDataCollector(
        output_dir="lidar_data_test",  # Using new folder to avoid lock
        scene_name="scene_lidar_drone",
        compute_features=True,  # Set to False for faster collection without features
        clear_existing=False  # Set to True to clear data before run
    )

    try:
        # Connect to simulation environment
        client.connect()

        # Create a World object to interact with the sim world and load a scene
        world = World(client, "scene_lidar_drone.jsonc", delay_after_load_sec=2)

        # Create a Drone object to interact with a drone in the loaded sim world
        drone = Drone(client, world, "Drone1")

        # Pass drone reference to collector for state information
        collector.set_drone(drone)

        # Setup visualization if enabled
        if ENABLE_VISUALIZATION:
            # Subscribe to chase camera sensor
            chase_cam_window = "ChaseCam"
            image_display.add_chase_cam(chase_cam_window)
            client.subscribe(
                drone.sensors["Chase"]["scene_camera"],
                lambda _, chase: image_display.receive(chase, chase_cam_window),
            )

            # Subscribe to the Drone's sensors with a callback to receive the sensor data
            rgb_name = "RGB-Image"
            image_display.add_image(rgb_name, subwin_idx=0)
            client.subscribe(
                drone.sensors["DownCamera"]["scene_camera"],
                lambda _, rgb: image_display.receive(rgb, rgb_name),
            )

            depth_name = "Depth-Image"
            image_display.add_image(depth_name, subwin_idx=1)
            client.subscribe(
                drone.sensors["DownCamera"]["depth_camera"],
                lambda _, depth: image_display.receive(depth, depth_name),
            )

            image_display.start()

            # Subscribe to LiDAR with visualization
            client.subscribe(
                drone.sensors["lidar1"]["lidar"],
                lambda _, lidar: lidar_display.receive(lidar),
            )

            lidar_display.start()

        # Subscribe to LiDAR with data collection (always enabled)
        client.subscribe(
            drone.sensors["lidar1"]["lidar"],
            collector.process_and_store,
        )

        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()

        # Fly the drone around the scene - then return to start
        projectairsim_log().info("Move up")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-3.0, duration=4.0
        )
        await move_task

        projectairsim_log().info("Move north")
        move_task = await drone.move_by_velocity_async(
            v_north=4.0, v_east=0.0, v_down=0.0, duration=12.0
        )
        await move_task

        projectairsim_log().info("Move north-east")
        move_task = await drone.move_by_velocity_async(
            v_north=4.0, v_east=4.0, v_down=0.0, duration=8.0
        )
        await move_task

        projectairsim_log().info("Move north")
        move_task = await drone.move_by_velocity_async(
            v_north=4.0, v_east=0.0, v_down=0.0, duration=3.0
        )
        await move_task

        projectairsim_log().info("Move down")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=3.0, duration=4.0
        )
        await move_task

        # === RETURN TO START ===
        projectairsim_log().info("Returning to starting position...")

        # Reverse the moves in correct order: up, reverse north movements, then land
        projectairsim_log().info("Move up (take off again)")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-3.0, duration=4.0
        )
        await move_task

        projectairsim_log().info("Move south (reverse last north)")
        move_task = await drone.move_by_velocity_async(
            v_north=-4.0, v_east=0.0, v_down=0.0, duration=3.0
        )
        await move_task

        projectairsim_log().info("Move south-west (reverse north-east)")
        move_task = await drone.move_by_velocity_async(
            v_north=-4.0, v_east=-4.0, v_down=0.0, duration=8.0
        )
        await move_task

        projectairsim_log().info("Move south (reverse first north)")
        move_task = await drone.move_by_velocity_async(
            v_north=-4.0, v_east=0.0, v_down=0.0, duration=12.0
        )
        await move_task

        projectairsim_log().info("Move down (land at start)")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=3.0, duration=4.0
        )
        await move_task

        projectairsim_log().info("Returned to starting position")

        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()

        # Save collection summary
        collector.save_summary()

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()

        # Stop displays if they were started
        if image_display:
            image_display.stop()
        if lidar_display:
            lidar_display.stop()


if __name__ == "__main__":
    asyncio.run(main())

