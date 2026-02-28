"""
Enhanced LiDAR Data Collector with RGB-Based Material Properties

Combines CPU LiDAR geometry with RGB camera appearance to derive material properties:
- Uses CPU LiDAR for accurate 3D point cloud (360° coverage)
- Projects points onto RGB camera to extract color information
- Derives material-like features from RGB (roughness, specular, metallic heuristics)
- Packages data in same format as lidar_kush.py for AI model compatibility

This replaces GPU LiDAR intensity with RGB-derived material properties.
"""

import asyncio
import threading
import numpy as np
from pathlib import Path
import time
import cv2
from sklearn.neighbors import BallTree
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, quaternion_to_rpy, unpack_image
from projectairsim.types import ImageType


class ColorMaterialLookup:
    """
    Maps surface colors (RGB) to physically-based material properties via
    nearest-neighbour search in Euclidean RGB space.

    Material table uses the scene's exact sRGB hex colours (alpha byte stripped).
    UE5 editor values above 1.0 (e.g. specular=2.0) are clamped to 1.0 here.
    Concrete specular=None → 0.1 (physical minimum for dielectrics).
    """

    DEFAULT_MATERIALS = {
        # label 0
        "asphalt":  {"rgb": (59,  59,  59),  "roughness": 0.2, "specular": 0.2, "metallic": 0.2},
        # label 1 — hex E1E1E1; specular originally None → 0.1
        "concrete": {"rgb": (225, 225, 225), "roughness": 0.0, "specular": 0.1, "metallic": 0.0},
        # label 2 — hex 1900E1 = (25, 0, 225); UE5 specular=2.0/metallic=2.0 → clamped to 1.0
        "metal":    {"rgb": (25,  0,   225), "roughness": 0.2, "specular": 1.0, "metallic": 1.0},
        # label 3 — hex 54762D
        "greenery": {"rgb": (84,  118, 45),  "roughness": 0.8, "specular": 0.2, "metallic": 0.0},
    }

    # Gaussian noise sigmas per property (applied per-point)
    NOISE_SIGMA = {"roughness": 0.04, "specular": 0.03, "metallic": 0.03}

    # Euclidean RGB distance threshold; beyond this → "unknown" (label -1)
    MAX_COLOR_DISTANCE = 60.0

    def __init__(self, material_table=None, noise_level=1.0):
        """
        material_table : dict matching DEFAULT_MATERIALS schema, or None to use defaults.
        noise_level    : scalar multiplier on NOISE_SIGMA. 0.0 = deterministic, 1.0 = default.
        """
        table = material_table if material_table is not None else self.DEFAULT_MATERIALS
        self._names          = list(table.keys())
        self._table_rgb      = np.array([e["rgb"]       for e in table.values()], dtype=np.float32)
        self._table_roughness = np.array([e["roughness"] for e in table.values()], dtype=np.float32)
        self._table_specular  = np.array([e["specular"]  for e in table.values()], dtype=np.float32)
        self._table_metallic  = np.array([e["metallic"]  for e in table.values()], dtype=np.float32)
        self._noise_level = float(noise_level)

    def lookup_batch(self, rgb_batch):
        """
        Fully vectorised nearest-neighbour lookup for a batch of RGB colours.

        Parameters
        ----------
        rgb_batch : (N, 3) uint8

        Returns
        -------
        roughness      : (N,) float32 — table value + per-point Gaussian noise
        specular       : (N,) float32
        metallic       : (N,) float32
        material_label : (N,) int32  — 0-based table index, -1 if beyond MAX_COLOR_DISTANCE
        has_material   : (N,) bool
        """
        n = len(rgb_batch)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool)

        # (N, M) distance matrix — fully vectorised
        rgb_f = rgb_batch.astype(np.float32)                          # (N, 3)
        diffs = rgb_f[:, None, :] - self._table_rgb[None, :, :]       # (N, M, 3)
        dists = np.linalg.norm(diffs, axis=2)                         # (N, M)
        best_idx  = np.argmin(dists, axis=1)                          # (N,)
        best_dist = dists[np.arange(n), best_idx]                     # (N,)

        has_material = best_dist < self.MAX_COLOR_DISTANCE

        # Pull base values from table
        base_r = self._table_roughness[best_idx]
        base_s = self._table_specular[best_idx]
        base_m = self._table_metallic[best_idx]

        # Add vectorised per-point Gaussian noise
        nl = self._noise_level
        roughness = np.clip(base_r + np.random.normal(0, nl * self.NOISE_SIGMA["roughness"], n), 0.0, 1.0).astype(np.float32)
        specular  = np.clip(base_s + np.random.normal(0, nl * self.NOISE_SIGMA["specular"],  n), 0.0, 1.0).astype(np.float32)
        metallic  = np.clip(base_m + np.random.normal(0, nl * self.NOISE_SIGMA["metallic"],  n), 0.0, 1.0).astype(np.float32)

        # Zero-out unknown points so they don't pollute training
        roughness[~has_material] = 0.0
        specular[~has_material]  = 0.0
        metallic[~has_material]  = 0.0

        label = np.where(has_material, best_idx, -1).astype(np.int32)
        return roughness, specular, metallic, label, has_material

    @property
    def material_names(self):
        """Ordered list of material names; index matches label returned by lookup_batch."""
        return list(self._names)


class RGBLidarDataCollector:
    """
    Collects LiDAR data enhanced with RGB-based material properties.
    Output format identical to lidar_kush.py for edge AI compatibility.
    """

    def __init__(self, drone, output_dir="lidar_data_rgb", scene_name="scene_lidar_drone",
                 camera_name="DownCamera", clear_existing=True,
                 noise_level=1.0, material_table=None):
        """
        noise_level    : Gaussian noise multiplier on material properties (0=none, 1=default).
        material_table : Custom colour→material dict (same schema as ColorMaterialLookup.DEFAULT_MATERIALS).
                         Pass None to use the built-in scene-specific table.
        """
        self.drone = drone
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.camera_name = camera_name
        self.material_lookup = ColorMaterialLookup(material_table, noise_level)
        
        if clear_existing and self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
            projectairsim_log().info(f"Cleared existing data from {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scan_count = 0
        self.start_time = time.time()
        
        # Camera parameters (DownCamera from robot config)
        self.image_width = 400
        self.image_height = 225
        self.fov_deg = 90
        self.fov_rad = np.deg2rad(self.fov_deg)
        self.focal_length = (self.image_width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        # Camera extrinsics: DownCamera at (0, 0, 0) pointing down (rpy-deg: 0 -90 0).
        # With pitch=-90 the optical axis aligns with body +Z (Down in NED).
        # For OpenCV pinhole: cam_X=body_Y(East), cam_Y=body_-X(South), cam_Z=body_Z(Down).
        # Verified: point [0,0,d] below drone → z_cam=d > 0 (in front of camera).
        self.drone_to_camera_rot = np.array([
            [ 0,  1,  0],   # cam X  = body Y  (East)
            [-1,  0,  0],   # cam Y  = body -X (South = "image-down" for downward cam)
            [ 0,  0,  1],   # cam Z  = body Z  (Down = optical axis depth)
        ], dtype=np.float64)
        
        # Latest RGB image + drone pose at capture time (updated by background task)
        self._rgb_lock = threading.Lock()
        self._latest_rgb_image = None
        self._latest_rgb_drone_quat = None  # (w,x,y,z) quaternion at image capture time
        self._latest_drone_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._latest_drone_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._shutdown = False  # Flag to stop RGB update loop cleanly
        self._last_state_warning_ts = 0.0

        # Movement tracking (set externally by execute_flight_pattern)
        self.current_movement_id = 0
        self.total_movements = 0

        # Video recording — one MP4 per flight, written in _update_rgb_loop
        self._video_writer: cv2.VideoWriter | None = None
        self._video_fps = 10  # matches _update_rgb_loop sleep of 0.1 s
        video_path = self.output_dir / "flight_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self._video_fps,
            (self.image_width, self.image_height),
        )
        if not self._video_writer.isOpened():
            projectairsim_log().warning(f"VideoWriter could not open {video_path}; video will not be saved")
            self._video_writer = None
        else:
            projectairsim_log().info(f"  Video: {video_path}")
        
        projectairsim_log().info(f"RGB+LiDAR data collector initialized")
        projectairsim_log().info(f"  Output: {self.output_dir.absolute()}")
        projectairsim_log().info(f"  Camera: {camera_name} ({self.image_width}x{self.image_height}, {self.fov_deg}°))")
    
    def _rpy_to_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg):
        """Convert roll-pitch-yaw to rotation matrix"""
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def lidar_callback(self, topic, lidar_data):
        """
        Process LiDAR scan immediately when received (callback-driven).
        This is much faster than polling - processes at LiDAR rate (~10Hz).
        """
        if lidar_data is None:
            return
        
        # Process synchronously (like lidar_kush.py does)
        self._process_scan(lidar_data)
    
    async def _update_rgb_loop(self):
        """Background task to update RGB image + drone pose periodically"""
        while not self._shutdown:
            try:
                images = self.drone.get_images(self.camera_name, [ImageType.SCENE])
                if ImageType.SCENE in images:
                    rgb_image = unpack_image(images[ImageType.SCENE])
                    # Capture drone state at the same moment as the image
                    drone_quat = None
                    drone_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    try:
                        state = self.drone.get_ground_truth_kinematics() or {}
                        pose = state.get("pose") or {}
                        q = pose.get("orientation") or {}
                        position = pose.get("position") or {}
                        linear = (state.get("twist") or {}).get("linear") or {}
                        if q:
                            drone_quat = np.array([
                                float(q.get("w", 1.0)), float(q.get("x", 0.0)),
                                float(q.get("y", 0.0)), float(q.get("z", 0.0)),
                            ], dtype=np.float64)
                        if position:
                            drone_pos = np.array([
                                float(position.get("x", 0.0)),
                                float(position.get("y", 0.0)),
                                float(position.get("z", 0.0)),
                            ], dtype=np.float32)
                        if linear:
                            drone_vel = np.array([
                                float(linear.get("x", 0.0)),
                                float(linear.get("y", 0.0)),
                                float(linear.get("z", 0.0)),
                            ], dtype=np.float32)
                    except Exception:
                        pass
                    with self._rgb_lock:
                        self._latest_rgb_image = rgb_image
                        self._latest_rgb_drone_quat = drone_quat
                        self._latest_drone_position = drone_pos
                        self._latest_drone_velocity = drone_vel

                    # Write frame to video (OpenCV expects BGR)
                    if self._video_writer is not None:
                        bgr_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                        self._video_writer.write(bgr_frame)

            except asyncio.CancelledError:
                # Task is being shut down externally — honor it
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Transient error (e.g. move command interrupted the request) — keep looping
                if not self._shutdown:
                    projectairsim_log().debug(f"RGB image fetch skipped: {e}")

            await asyncio.sleep(0.1)  # Update at 10Hz (matches LiDAR scan rate)

        # Release video writer when loop exits (shutdown or cancel)
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            projectairsim_log().info("Video recording finalized")

    def _quat_to_rotation_matrix(self, q):
        """Convert (w,x,y,z) quaternion to 3x3 rotation matrix (body→world)"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _project_points_to_camera(self, points_drone_frame, drone_quat=None):
        """
        Project 3D points (body frame) onto 2D camera plane.

        If drone_quat is provided (quaternion at image-capture time), the drone's
        tilt at that moment is factored out so LiDAR body-frame points align with
        the image even when the drone is banked/pitched during flight.
        """
        if drone_quat is not None:
            # R_body_to_world at image capture time
            R_bw = self._quat_to_rotation_matrix(drone_quat)
            # Rotate points from current body frame → world → image-capture body frame
            # (equivalent to undoing the drone's orientation)
            # world_pts = R_bw @ pts  →  image_body_pts = R_bw.T @ world_pts = pts
            # This compensates for tilt mismatch between LiDAR scan and RGB capture.
            pts = (R_bw.T @ (R_bw @ points_drone_frame.T)).T  # simplifies to identity
            # Actually: if LiDAR pts are in body frame and image was also captured
            # in body frame, no correction is needed.  The real correction is only
            # required when the image was captured at a DIFFERENT drone orientation
            # than the current LiDAR scan.  We store the quat at capture time;
            # at LiDAR callback time we read the current quat separately.
            pts = points_drone_frame  # kept for future extension (see note below)
        else:
            pts = points_drone_frame

        points_camera_frame = (self.drone_to_camera_rot @ pts.T).T
        valid_depth = points_camera_frame[:, 2] > 0.1

        # Avoid division by zero for points near the depth=0 plane
        safe_z = np.where(valid_depth, points_camera_frame[:, 2], 1.0)
        u = self.focal_length * (points_camera_frame[:, 0] / safe_z) + self.cx
        v = self.focal_length * (points_camera_frame[:, 1] / safe_z) + self.cy

        valid_u = (u >= 0) & (u < self.image_width)
        valid_v = (v >= 0) & (v < self.image_height)
        valid_mask = valid_depth & valid_u & valid_v

        pixel_coords = np.column_stack([u, v])
        return pixel_coords, valid_mask
    
    def _extract_rgb_material_features(self, rgb_image, points, pixel_coords, valid_mask):
        """
        Project LiDAR points onto the RGB camera, read surface colour, and look up
        material properties (roughness, specular, metallic) from the ColorMaterialLookup
        table with per-point Gaussian noise injection.

        Returns (for ALL N points, zeroed where camera coverage is absent):
            rgb_values       : (N, 3) uint8   — sampled sRGB colour
            material_roughness: (N,) float32  — from table + noise
            material_specular : (N,) float32  — from table + noise
            material_metallic : (N,) float32  — from table + noise
            material_label   : (N,) int32     — table index (0-based), -1 if unknown/no-coverage
            has_material     : (N,) bool      — True where a colour match was found
            material_intensity: (N,) float32  — derived from shader formula, [0, 1]
        """
        n_points = len(points)

        rgb_values        = np.zeros((n_points, 3), dtype=np.uint8)
        mat_roughness     = np.zeros(n_points, dtype=np.float32)
        mat_specular      = np.zeros(n_points, dtype=np.float32)
        mat_metallic      = np.zeros(n_points, dtype=np.float32)
        mat_label         = np.full(n_points, -1, dtype=np.int32)
        has_material      = np.zeros(n_points, dtype=bool)
        mat_intensity     = np.zeros(n_points, dtype=np.float32)

        if not np.any(valid_mask):
            return rgb_values, mat_roughness, mat_specular, mat_metallic, mat_label, has_material, mat_intensity

        # Sample RGB at the projected pixel coordinates
        valid_pixels = pixel_coords[valid_mask].astype(int)
        u = np.clip(valid_pixels[:, 0], 0, self.image_width - 1)
        v = np.clip(valid_pixels[:, 1], 0, self.image_height - 1)
        rgb_valid = rgb_image[v, u]           # (M, 3) uint8 for M projected points
        rgb_values[valid_mask] = rgb_valid

        # Lookup material properties with noise — fully vectorised
        r, s, m, lbl, hit = self.material_lookup.lookup_batch(rgb_valid)
        mat_roughness[valid_mask] = r
        mat_specular[valid_mask]  = s
        mat_metallic[valid_mask]  = m
        mat_label[valid_mask]     = lbl
        has_material[valid_mask]  = hit

        # Derive a single intensity scalar matching the GPU shader formula
        # (LidarIntensityPS.usf): 50*R*angle + 700*S*(1-angle) + 300*M,  /1000 → [0,1]
        # Simplified: angle_similarity = 0.5 (no normal estimate needed)
        mat_intensity[valid_mask] = (
            50.0  * r * 0.5 +
            700.0 * s * 0.5 +
            300.0 * m
        ) / 1000.0

        return rgb_values, mat_roughness, mat_specular, mat_metallic, mat_label, has_material, mat_intensity
    
    def _process_scan(self, lidar_data):
        """
        Process one scan: merge CPU LiDAR + RGB camera.
        Called automatically by lidar_callback when new data arrives.
        Saves in same format as lidar_kush.py for AI compatibility.
        """
        start_time = time.time()
        timestamp = start_time - self.start_time
        
        # === PARSE CPU LIDAR DATA ===
        points = np.array(lidar_data["point_cloud"], dtype=np.float32).reshape(-1, 3)
        segmentation = np.array(lidar_data["segmentation_cloud"], dtype=np.int32)
        
        if len(points) == 0:
            return  # Skip empty scans
        
        # === GET RGB IMAGE + pose at capture time (from cache) ===
        with self._rgb_lock:
            rgb_image = self._latest_rgb_image
            rgb_drone_quat = self._latest_rgb_drone_quat

        if rgb_image is None:
            # No RGB image yet, skip this scan
            return

        # === PROJECT POINTS ONTO CAMERA ===
        pixel_coords, valid_mask = self._project_points_to_camera(points, rgb_drone_quat)
        
        # === EXTRACT MATERIAL FEATURES VIA COLOR LOOKUP TABLE ===
        rgb_values, mat_roughness, mat_specular, mat_metallic, mat_label, has_material, mat_intensity = \
            self._extract_rgb_material_features(rgb_image, points, pixel_coords, valid_mask)

        # === COMPUTE GEOMETRY-BASED FEATURES (CPU LiDAR only) ===
        distance = np.linalg.norm(points.astype(np.float64), axis=1).astype(np.float32)
        distance = np.maximum(distance, 0.1)

        # Use material intensity (shader-formula-derived) on 0-1000 scale for consistency
        intensity = mat_intensity * 1000.0

        # Normalized intensity (distance-corrected reflectivity)
        normalized_intensity = (intensity.astype(np.float64) * (distance.astype(np.float64) ** 2)).astype(np.float32)

        # Geometric roughness from point cloud
        roughness_geom = self._compute_surface_roughness(points, k=10)

        # Combined roughness: worst-case of geometric and material table
        roughness = np.maximum(roughness_geom, mat_roughness)

        # Backscatter coefficient
        backscatter = normalized_intensity / (1.0 + roughness + 1e-6)

        # Intensity variance (local, driven by material_intensity noise)
        intensity_variance = self._compute_intensity_variance(points, intensity, k=10)

        # Viewing angle (simplified)
        viewing_angle = np.zeros(len(points), dtype=np.float32)
        
        # === GET DRONE STATE from cache (populated by _update_rgb_loop on asyncio thread) ===
        # We do NOT call get_ground_truth_kinematics() here because this runs on the NNG
        # subscriber callback thread and making a synchronous RPC from it is not supported.
        with self._rgb_lock:
            drone_position = self._latest_drone_position.copy()
            drone_velocity = self._latest_drone_velocity.copy()
            cached_quat = self._latest_rgb_drone_quat

        if cached_quat is not None:
            q_w, q_x, q_y, q_z = float(cached_quat[0]), float(cached_quat[1]), float(cached_quat[2]), float(cached_quat[3])
        else:
            q_w, q_x, q_y, q_z = 1.0, 0.0, 0.0, 0.0
        drone_orientation_quat = np.array([q_w, q_x, q_y, q_z], dtype=np.float32)
        roll, pitch, yaw = quaternion_to_rpy(q_w, q_x, q_y, q_z)
        drone_orientation_euler = np.array([roll, pitch, yaw], dtype=np.float32)
        
        # === SAVE NPZ ===
        filename = self.output_dir / f"scan_{self.scan_count:04d}.npz"
        np.savez_compressed(
            filename,
            # === RAW DATA ===
            points=points,
            intensity=intensity,          # shader-formula intensity on [0, 1000] scale
            segmentation=segmentation,
            timestamp=timestamp,
            scan_id=self.scan_count,
            movement_id=self.current_movement_id,
            total_movements=self.total_movements,

            # === GEOMETRY FEATURES (CPU LiDAR) ===
            distance=distance,
            normalized_intensity=normalized_intensity,
            roughness=roughness,          # max(geometric, material_roughness)
            backscatter=backscatter,
            intensity_variance=intensity_variance,
            viewing_angle=viewing_angle,

            # === MATERIAL FEATURES (color lookup table + noise) ===
            rgb=rgb_values,               # (N, 3) uint8 sampled surface colour
            material_roughness=mat_roughness,
            material_specular=mat_specular,
            material_metallic=mat_metallic,
            material_intensity=mat_intensity,   # [0, 1] shader-formula intensity
            material_label=mat_label,     # int32: 0=asphalt,1=concrete,2=metal,3=greenery,-1=unknown
            has_material=has_material,    # bool: True where colour match found

            # === METADATA ===
            scene_name=self.scene_name,
            drone_position=drone_position,
            drone_velocity=drone_velocity,
            drone_orientation_quat=drone_orientation_quat,
            drone_orientation_euler=drone_orientation_euler,
        )

        process_time = time.time() - start_time
        n_with_mat = int(has_material.sum())
        pct_mat = 100.0 * n_with_mat / len(points)
        
        projectairsim_log().info(
            f"Scan {self.scan_count}: {len(points)} pts, {n_with_mat} material-matched "
            f"({pct_mat:.1f}%) -> {filename.name} ({process_time:.2f}s)"
        )
        
        self.scan_count += 1
    
    def _compute_surface_roughness(self, points, k=10):
        """Geometric roughness from point cloud (same as lidar_kush.py)"""
        if len(points) < k:
            return np.zeros(len(points), dtype=np.float32)
        
        tree = BallTree(points, leaf_size=40)
        _, indices = tree.query(points, k=min(k, len(points)))
        
        roughness = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            neighbors = points[indices[i]].astype(np.float64)
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                roughness[i] = eigenvalues[0] / (eigenvalues[2] + 1e-6)
            except:
                pass
        
        return roughness
    
    def _compute_intensity_variance(self, points, intensity, k=10):
        """Local intensity variance (same as lidar_kush.py)"""
        if len(points) < k:
            return np.zeros(len(points), dtype=np.float32)
        
        tree = BallTree(points, leaf_size=40)
        _, indices = tree.query(points, k=min(k, len(points)))
        
        intensity_var = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            neighbor_intensities = intensity[indices[i]]
            intensity_var[i] = np.var(neighbor_intensities)
        
        return intensity_var


async def main():
    """Test RGB+LiDAR data collection"""
    
    print("="*60)
    print("RGB+LiDAR Data Collector (AI-Compatible Format)")
    print("="*60)
    
    client = ProjectAirSimClient()
    sim_config_dir = str(Path(__file__).parent / "sim_config")

    try:
        client.connect()

        world = World(client, "scene_lidar_drone.jsonc", delay_after_load_sec=2,
                      sim_config_path=sim_config_dir)
        drone = Drone(client, world, "Drone1")

        # Create collector
        collector = RGBLidarDataCollector(drone, output_dir="lidar_data_rgb")

        # Start RGB image update loop in background
        rgb_update_task = asyncio.create_task(collector._update_rgb_loop())

    except Exception as err:
        print(f"Setup failed: {err}")
        client.disconnect()
        return

    try:
        # Subscribe to CPU LiDAR
        client.subscribe(drone.sensors["cpu_lidar1"]["lidar"], collector.lidar_callback)
        
        # Takeoff
        print("\nTaking off...")
        drone.enable_api_control()
        drone.arm()
        await drone.takeoff_async()
        
        # Move to altitude
        print("Climbing to 10m...")
        cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
        await drone.move_to_position_async(north=cur_pos["x"], east=cur_pos["y"], down=-10.0, velocity=2.0)
        
        # Wait for LiDAR to stabilize
        print("Stabilizing...")
        await asyncio.sleep(2.0)
        
        # Collect scans automatically via callback (NO manual delays!)
        print("\nCollecting 10 scans (callback-driven, ~10Hz)...")
        print("Scans will be saved automatically as LiDAR data arrives...")
        
        # Just wait for scans to accumulate (collector processes them automatically)
        target_scans = 10
        while collector.scan_count < target_scans:
            await asyncio.sleep(0.1)  # Check every 100ms
            if collector.scan_count > 0 and collector.scan_count % 5 == 0:
                print(f"  Progress: {collector.scan_count}/{target_scans} scans collected")
        
        print(f"\n✓ Collected {collector.scan_count} scans!")
        
        print("\n" + "="*60)
        print(f"Data collection complete! Saved to: {collector.output_dir}")
        print("="*60)
        
        # Stop RGB update loop
        collector._shutdown = True
        rgb_update_task.cancel()
        try:
            await rgb_update_task
        except asyncio.CancelledError:
            pass
        
        # Land using velocity (like other scripts do)
        print("\nLanding...")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=3.0, duration=4.0
        )
        await move_task
        
        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()
        
        print("Done!")
        
    except Exception as err:
        print(f"Exception occurred: {err}")
        
    finally:
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
