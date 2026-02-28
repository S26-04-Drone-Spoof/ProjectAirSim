"""
RGB LiDAR Dataset Generator
Uses RGBLidarDataCollector to generate training datasets with RGB-derived material properties.
Drop-in replacement for lidar_dataset_generator.py but uses RGB instead of GPU LiDAR.
"""

import asyncio
import numpy as np
from pathlib import Path

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log

import sys
sys.path.insert(0, str(Path(__file__).parent))
from rgb_lidar_collector import RGBLidarDataCollector


# ============================================================================
# FLIGHT PATTERN DEFINITIONS (identical to lidar_dataset_generator.py)
# ============================================================================

def generate_random_flight_pattern(seed=None, num_segments=8, max_velocity=5.0, max_altitude=17.0):
    if seed is not None:
        np.random.seed(seed)

    pattern = []

    takeoff_altitude = max_altitude  # Always fly at max altitude to clear structures
    takeoff_duration = takeoff_altitude / 3.0
    pattern.append((0.0, 0.0, -3.0, takeoff_duration))

    for _ in range(num_segments):
        target_distance = np.random.uniform(10.0, 40.0)
        direction_angle = np.random.uniform(0, 2 * np.pi)
        horizontal_speed = np.random.uniform(2.0, max_velocity)
        v_north = horizontal_speed * np.cos(direction_angle)
        v_east = horizontal_speed * np.sin(direction_angle)
        duration = target_distance / horizontal_speed
        pattern.append((v_north, v_east, 0.0, duration))

    landing_duration = takeoff_altitude / 3.0
    pattern.append((0.0, 0.0, 3.0, landing_duration))

    return pattern


def get_flight_patterns():
    patterns = {
        "forward_scan": [
            (0.0, 0.0, -3.0, 4.0),
            (4.0, 0.0, 0.0, 12.0),
            (4.0, 4.0, 0.0, 8.0),
            (4.0, 0.0, 0.0, 3.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "backward_scan": [
            (0.0, 0.0, -3.0, 4.0),
            (-4.0, 0.0, 0.0, 12.0),
            (-4.0, -4.0, 0.0, 8.0),
            (-4.0, 0.0, 0.0, 3.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "circle_clockwise": [
            (0.0, 0.0, -3.0, 4.0),
            (3.0, 0.0, 0.0, 5.0),
            (3.0, 3.0, 0.0, 5.0),
            (0.0, 3.0, 0.0, 5.0),
            (-3.0, 3.0, 0.0, 5.0),
            (-3.0, 0.0, 0.0, 5.0),
            (-3.0, -3.0, 0.0, 5.0),
            (0.0, -3.0, 0.0, 5.0),
            (3.0, -3.0, 0.0, 5.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "high_altitude_scan": [
            (0.0, 0.0, -5.0, 6.0),
            (3.0, 0.0, 0.0, 10.0),
            (0.0, 3.0, 0.0, 8.0),
            (-3.0, 0.0, 0.0, 10.0),
            (0.0, -3.0, 0.0, 8.0),
            (0.0, 0.0, 5.0, 6.0),
        ],
        "low_altitude_scan": [
            (0.0, 0.0, -2.0, 3.0),
            (2.0, 0.0, 0.0, 8.0),
            (0.0, 2.0, 0.0, 6.0),
            (-2.0, 0.0, 0.0, 8.0),
            (0.0, -2.0, 0.0, 6.0),
            (0.0, 0.0, 2.0, 3.0),
        ],
        "diagonal_sweep": [
            (0.0, 0.0, -3.0, 4.0),
            (4.0, 4.0, 0.0, 10.0),
            (-4.0, 4.0, 0.0, 10.0),
            (-4.0, -4.0, 0.0, 10.0),
            (4.0, -4.0, 0.0, 10.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "spiral_up": [
            (2.0, 0.0, -1.5, 4.0),
            (2.0, 2.0, -1.5, 4.0),
            (0.0, 2.0, -1.5, 4.0),
            (-2.0, 2.0, -1.5, 4.0),
            (-2.0, 0.0, -1.5, 4.0),
            (0.0, 0.0, 6.0, 6.0),
        ],
        "figure_eight": [
            (0.0, 0.0, -3.0, 4.0),
            (3.0, 2.0, 0.0, 4.0),
            (2.0, 3.0, 0.0, 4.0),
            (-2.0, 3.0, 0.0, 4.0),
            (-3.0, 2.0, 0.0, 4.0),
            (0.0, -2.0, 0.0, 4.0),
            (3.0, -2.0, 0.0, 4.0),
            (2.0, -3.0, 0.0, 4.0),
            (-2.0, -3.0, 0.0, 4.0),
            (-3.0, -2.0, 0.0, 4.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "west_sweep": [
            (0.0, 0.0, -3.0, 4.0),
            (0.0, -4.0, 0.0, 10.0),
            (3.0, -3.0, 0.0, 8.0),
            (0.0, -4.0, 0.0, 8.0),
            (-3.0, -3.0, 0.0, 8.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "northwest_arc": [
            (0.0, 0.0, -3.0, 4.0),
            (4.0, -2.0, 0.0, 10.0),
            (2.0, -4.0, 0.0, 8.0),
            (-2.0, -4.0, 0.0, 8.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
        "left_circle": [
            (0.0, 0.0, -3.0, 4.0),
            (-3.0, 0.0, 0.0, 5.0),
            (-3.0, -3.0, 0.0, 5.0),
            (0.0, -3.0, 0.0, 5.0),
            (3.0, -3.0, 0.0, 5.0),
            (3.0, 0.0, 0.0, 5.0),
            (0.0, 0.0, 3.0, 4.0),
        ],
    }

    NUM_RANDOM_PATTERNS = 30
    for i in range(NUM_RANDOM_PATTERNS):
        patterns[f"random_{i:02d}"] = generate_random_flight_pattern(
            seed=i,
            num_segments=8,
            max_velocity=5.0,
            max_altitude=15.0,
        )

    return patterns


# ============================================================================
# FLIGHT EXECUTION
# ============================================================================

async def execute_flight_pattern(drone, pattern_commands, collector=None):
    total = len(pattern_commands)
    if collector is not None:
        collector.total_movements = total
    for idx, (v_north, v_east, v_down, duration) in enumerate(pattern_commands):
        if collector is not None:
            collector.current_movement_id = idx
        direction = "up" if v_down < 0 else "down" if v_down > 0 else "level"
        projectairsim_log().info(
            f"  Movement {idx+1}/{total}: "
            f"N={v_north:.1f} E={v_east:.1f} D={v_down:.1f} ({direction}) "
            f"for {duration:.1f}s"
        )
        move_task = await drone.move_by_velocity_async(
            v_north=v_north, v_east=v_east, v_down=v_down, duration=duration
        )
        await move_task


# ============================================================================
# DATASET GENERATION
# ============================================================================

async def generate_dataset():
    SCENE_FILE = "scene_lidar_drone.jsonc"
    DATASET_ROOT = Path("lidar_spoofed_foliage_170_028_095_asphalt_148_241_dataset")
    MAX_SCENARIOS = 30

    DATASET_ROOT.mkdir(exist_ok=True)
    projectairsim_log().info(f"Dataset will be saved to: {DATASET_ROOT.absolute()}")

    flight_patterns = get_flight_patterns()
    projectairsim_log().info(f"Will execute up to {MAX_SCENARIOS} of {len(flight_patterns)} flight patterns")

    script_dir = Path(__file__).parent
    sim_config_dir = script_dir / "sim_config"

    client = ProjectAirSimClient()

    try:
        client.connect()
        projectairsim_log().info("Connected to simulation")

        flight_patterns_list = list(flight_patterns.items())[:MAX_SCENARIOS]
        for scenario_id, (pattern_name, pattern_commands) in enumerate(flight_patterns_list):
            projectairsim_log().info("")
            projectairsim_log().info("=" * 70)
            projectairsim_log().info(f"SCENARIO {scenario_id+1}/{len(flight_patterns_list)}: {pattern_name}")
            projectairsim_log().info("=" * 70)

            scenario_dir = DATASET_ROOT / f"scenario_{scenario_id:04d}_{pattern_name}"
            projectairsim_log().info(f"Output directory: {scenario_dir}")

            # Fresh world + drone each scenario — reloads scene, drone spawns at origin
            # Retry loop: load_scene has a race between its internal unsubscribe_all
            # and the background NNG subscriber thread; retry if that window is hit.
            projectairsim_log().info(f"Loading scene: {SCENE_FILE}")
            world = None
            for attempt in range(5):
                try:
                    world = World(
                        client,
                        SCENE_FILE,
                        delay_after_load_sec=2,
                        sim_config_path=str(sim_config_dir),
                    )
                    break
                except Exception as e:
                    if attempt < 4:
                        projectairsim_log().warning(
                            f"Scene load attempt {attempt+1} failed ({e}), retrying in 3s..."
                        )
                        await asyncio.sleep(3.0)
                    else:
                        raise
            drone = Drone(client, world, "Drone1")

            rgb_collector = RGBLidarDataCollector(
                drone,
                output_dir=str(scenario_dir),
                scene_name=SCENE_FILE,
                camera_name="DownCamera",
                clear_existing=False,
            )

            rgb_update_task = asyncio.create_task(rgb_collector._update_rgb_loop())

            client.subscribe(
                drone.sensors["cpu_lidar1"]["lidar"],
                rgb_collector.lidar_callback,
            )

            drone.enable_api_control()
            drone.arm()
            projectairsim_log().info("Drone armed and ready")

            # Brief settle before flight begins
            await asyncio.sleep(1.0)

            projectairsim_log().info(f"Executing flight pattern: {pattern_name}")
            await execute_flight_pattern(drone, pattern_commands)

            drone.disarm()
            drone.disable_api_control()
            projectairsim_log().info("Drone disarmed")

            rgb_collector._shutdown = True
            rgb_update_task.cancel()
            try:
                await rgb_update_task
            except asyncio.CancelledError:
                pass

            # Explicitly close all NNG subscriptions before the next scene reload.
            # This mirrors the TestBench pattern: unsubscribe_all() stops the background
            # subscriber thread so World()'s internal unsubscribe_all()+request() has
            # nothing racing against it.
            client.unsubscribe_all()

            # Brief pause for any in-flight message processing to finish
            await asyncio.sleep(2.0)

            projectairsim_log().info(
                f"Scenario {scenario_id+1} complete - "
                f"{rgb_collector.scan_count} scans saved to {scenario_dir}"
            )

        projectairsim_log().info("")
        projectairsim_log().info("=" * 70)
        projectairsim_log().info("DATASET GENERATION COMPLETE")
        projectairsim_log().info("=" * 70)
        projectairsim_log().info(f"Generated {len(flight_patterns_list)} scenarios")
        projectairsim_log().info(f"Dataset location: {DATASET_ROOT.absolute()}")

        total_scans = 0
        for subdir in sorted(DATASET_ROOT.iterdir()):
            if subdir.is_dir():
                npz_files = list(subdir.glob("scan_*.npz"))
                total_scans += len(npz_files)
                projectairsim_log().info(f"  {subdir.name:40s} - {len(npz_files):4d} scans")
        projectairsim_log().info(f"  {'TOTAL':40s} - {total_scans:4d} scans")

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        client.disconnect()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(generate_dataset())
