"""
GPU LiDAR Intensity Diagnostic Script

Checks raw GPU LiDAR output to diagnose why intensity values are extremely low.
Run this with the simulation running to inspect GPU intensity in real-time.

The GPU shader should output intensity in range [0.04 - 0.7] based on:
    Intensity = 0.05*Roughness*angle + 0.7*Specular*(1-angle) + 0.3*Metallic

If you're seeing values around 1e-06, possible causes:
1. GBuffer material properties are near-zero (check material setup in Unreal)
2. Intensity render target format issue (precision loss)
3. Shader output not being read correctly
4. Post-processing or gamma correction affecting values
"""

import asyncio
import numpy as np
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log


class GPUIntensityDiagnostic:
    def __init__(self):
        self.sample_count = 0
        self.intensity_samples = []
        
    def analyze_scan(self, topic, lidar_data):
        """Callback to analyze raw GPU LiDAR data"""
        if lidar_data is None:
            return
        
        pts = np.array(lidar_data["point_cloud"], dtype=np.float32).reshape(-1, 3)
        inty = np.array(lidar_data["intensity_cloud"], dtype=np.float32)
        
        # Filter to valid points only
        dist = np.linalg.norm(pts.astype(np.float64), axis=1).astype(np.float32)
        valid = (
            np.any(pts != 0.0, axis=1) &
            (dist > 0.5) &
            (dist < 500.0) &
            np.isfinite(inty)
        )
        
        pts_valid = pts[valid]
        inty_valid = inty[valid]
        
        if len(inty_valid) == 0:
            projectairsim_log().warning("No valid points in this scan!")
            return
        
        self.sample_count += 1
        self.intensity_samples.extend(inty_valid.tolist())
        
        # Print statistics for this scan
        print(f"\n=== Scan {self.sample_count} ===")
        print(f"Valid points: {len(inty_valid):,} / {len(inty):,}")
        print(f"Intensity stats:")
        print(f"  min   = {np.min(inty_valid):.6e}")
        print(f"  max   = {np.max(inty_valid):.6e}")
        print(f"  mean  = {np.mean(inty_valid):.6e}")
        print(f"  median= {np.median(inty_valid):.6e}")
        print(f"  std   = {np.std(inty_valid):.6e}")
        
        # Check for patterns
        zero_count = np.sum(inty_valid == 0.0)
        if zero_count > 0:
            print(f"  WARNING: {zero_count} zero intensity values ({100*zero_count/len(inty_valid):.1f}%)")
        
        # Check if values are in expected range
        expected_min = 0.01  # Conservative lower bound
        expected_max = 1.0   # Conservative upper bound
        
        if np.max(inty_valid) < expected_min:
            print(f"  ⚠️  ISSUE: All intensity values < {expected_min} (expected 0.04-0.7)")
            print(f"      This suggests GBuffer material properties are near-zero")
            print(f"      Check Unreal material setup (Roughness, Specular, Metallic)")
        elif np.max(inty_valid) > expected_max:
            print(f"  ⚠️  ISSUE: Some intensity values > {expected_max}")
            print(f"      Check shader coefficients or render target precision")
        else:
            print(f"  ✓ Intensity values in reasonable range")
        
        # Stop after 10 scans
        if self.sample_count >= 10:
            print("\n" + "="*60)
            print("OVERALL STATISTICS (10 scans)")
            print("="*60)
            all_inty = np.array(self.intensity_samples)
            print(f"Total points: {len(all_inty):,}")
            print(f"Intensity range: [{np.min(all_inty):.6e}, {np.max(all_inty):.6e}]")
            print(f"Mean: {np.mean(all_inty):.6e}")
            print(f"Median: {np.median(all_inty):.6e}")
            print(f"Std dev: {np.std(all_inty):.6e}")
            
            # Histogram
            print("\nIntensity histogram (10 bins):")
            hist, bins = np.histogram(all_inty, bins=10)
            for i in range(len(hist)):
                bar = "#" * int(50 * hist[i] / np.max(hist))
                print(f"  [{bins[i]:.2e}, {bins[i+1]:.2e}): {hist[i]:6d} {bar}")
            
            print("\n" + "="*60)
            print("RECOMMENDATIONS:")
            print("="*60)
            
            if np.max(all_inty) < 0.01:
                print("1. Check Unreal materials - Specular/Roughness/Metallic likely near zero")
                print("2. In Unreal Editor:")
                print("   - Select a surface material")
                print("   - Check Roughness (should be 0.3-0.9 for typical surfaces)")
                print("   - Check Specular (should be ~0.5 for non-metals)")
                print("   - Check Metallic (0=dielectric, 1=metal)")
                print("3. If using default materials, they may have very low reflectance")
            else:
                print("Intensity values look reasonable! GPU LiDAR is working correctly.")


async def main():
    client = ProjectAirSimClient()
    diagnostic = GPUIntensityDiagnostic()
    
    try:
        client.connect()
        projectairsim_log().info("Connected to simulation")
        
        # Load the world and create drone object
        world = World(client, "scene_lidar_drone.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        
        # Enable API control and arm the drone
        drone.enable_api_control()
        drone.arm()
        
        # Take off to 10 meters so downward camera can see ground properly
        projectairsim_log().info("Taking off to 10 meters altitude...")
        takeoff_task = await drone.takeoff_async()
        await takeoff_task
        
        # Move to 10m altitude (down=-10 in NED coordinates)
        cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
        move_task = await drone.move_to_position_async(
            north=cur_pos["x"], east=cur_pos["y"], down=-10.0, velocity=2.0
        )
        await move_task
        projectairsim_log().info("Hovering at 10 meters")
        
        # Subscribe to GPU LiDAR
        projectairsim_log().info("Subscribing to GPU LiDAR (lidar1)...")
        projectairsim_log().info("Will collect 10 scans and analyze intensity values...")
        
        client.subscribe(
            drone.sensors["lidar1"]["lidar"],
            diagnostic.analyze_scan
        )
        
        # Let it run for a while
        await asyncio.sleep(30)  # 30 seconds should be enough for 10 scans
        
    except KeyboardInterrupt:
        projectairsim_log().info("Interrupted by user")
    except Exception as e:
        projectairsim_log().error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()
        projectairsim_log().info("Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
