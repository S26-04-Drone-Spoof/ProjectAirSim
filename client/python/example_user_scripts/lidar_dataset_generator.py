"""
LiDAR Dataset Generator - Unspoofed Training Data
Flies multiple flight patterns in the same scene to generate diverse LiDAR data.
Each flight pattern is saved to a separate folder for organized dataset structure.
"""

import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log
from projectairsim.image_utils import ImageDisplay
from projectairsim.lidar_utils import LidarDisplay

from lidar_kush import LidarDataCollector


# ============================================================================
# FLIGHT PATTERN DEFINITIONS
# ============================================================================

def generate_random_flight_pattern(seed=None, num_segments=8, max_velocity=5.0, max_altitude=15.0):
    """
    Generate a random flight pattern from the starting point.

    Args:
        seed: Random seed for reproducibility (None for truly random)
        num_segments: Number of movement segments in the flight
        max_velocity: Maximum horizontal velocity (m/s)
        max_altitude: Maximum altitude above start (meters, positive = up)

    Returns:
        List of (v_north, v_east, v_down, duration) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    pattern = []

    # 1. Random takeoff (varying altitude)
    takeoff_altitude = np.random.uniform(2.0, max_altitude)
    takeoff_duration = takeoff_altitude / 3.0  # ~3 m/s ascent
    pattern.append((0.0, 0.0, -3.0, takeoff_duration))

    # 2. Random flight segments
    for _ in range(num_segments):
        # Random target travel distance (how far to move in this segment)
        target_distance = np.random.uniform(10.0, 40.0)  # meters

        # Random direction (angle in radians)
        direction_angle = np.random.uniform(0, 2 * np.pi)

        # Random speed (within limits)
        horizontal_speed = np.random.uniform(2.0, max_velocity)

        # Calculate velocities to achieve the direction at desired speed
        v_north = horizontal_speed * np.cos(direction_angle)
        v_east = horizontal_speed * np.sin(direction_angle)

        # Small random altitude change
        v_down = np.random.uniform(-1.0, 1.0)

        # Calculate duration needed: duration = distance / speed
        duration = target_distance / horizontal_speed

        pattern.append((v_north, v_east, v_down, duration))

    # 3. Landing
    landing_duration = takeoff_altitude / 3.0
    pattern.append((0.0, 0.0, 3.0, landing_duration))

    return pattern


def get_flight_patterns():
    """
    Define different flight patterns to view objects from various angles.
    Each pattern is a sequence of (v_north, v_east, v_down, duration) commands.

    Returns:
        dict: Flight pattern name -> list of movement commands
    """

    patterns = {
        "forward_scan": [
            # Takeoff
            (0.0, 0.0, -3.0, 4.0),   # Move up
            # Forward flight
            (4.0, 0.0, 0.0, 12.0),   # Move north
            (4.0, 4.0, 0.0, 8.0),    # Move north-east
            (4.0, 0.0, 0.0, 3.0),    # Move north
            # Land
            (0.0, 0.0, 3.0, 4.0),    # Move down
        ],

        "backward_scan": [
            # Takeoff
            (0.0, 0.0, -3.0, 4.0),   # Move up
            # Backward flight (reverse of forward)
            (-4.0, 0.0, 0.0, 12.0),  # Move south
            (-4.0, -4.0, 0.0, 8.0),  # Move south-west
            (-4.0, 0.0, 0.0, 3.0),   # Move south
            # Land
            (0.0, 0.0, 3.0, 4.0),    # Move down
        ],

        "circle_clockwise": [
            # Takeoff
            (0.0, 0.0, -3.0, 4.0),   # Move up
            # Circle pattern
            (3.0, 0.0, 0.0, 5.0),    # Move north
            (3.0, 3.0, 0.0, 5.0),    # Move north-east
            (0.0, 3.0, 0.0, 5.0),    # Move east
            (-3.0, 3.0, 0.0, 5.0),   # Move south-east
            (-3.0, 0.0, 0.0, 5.0),   # Move south
            (-3.0, -3.0, 0.0, 5.0),  # Move south-west
            (0.0, -3.0, 0.0, 5.0),   # Move west
            (3.0, -3.0, 0.0, 5.0),   # Move north-west
            # Land
            (0.0, 0.0, 3.0, 4.0),    # Move down
        ],

        "high_altitude_scan": [
            # High takeoff
            (0.0, 0.0, -5.0, 6.0),   # Move up higher
            # Survey pattern at altitude
            (3.0, 0.0, 0.0, 10.0),   # Move north
            (0.0, 3.0, 0.0, 8.0),    # Move east
            (-3.0, 0.0, 0.0, 10.0),  # Move south
            (0.0, -3.0, 0.0, 8.0),   # Move west
            # Land
            (0.0, 0.0, 5.0, 6.0),    # Move down
        ],

        "low_altitude_scan": [
            # Low takeoff
            (0.0, 0.0, -2.0, 3.0),   # Move up (lower)
            # Close inspection pattern
            (2.0, 0.0, 0.0, 8.0),    # Move north slowly
            (0.0, 2.0, 0.0, 6.0),    # Move east slowly
            (-2.0, 0.0, 0.0, 8.0),   # Move south slowly
            (0.0, -2.0, 0.0, 6.0),   # Move west slowly
            # Land
            (0.0, 0.0, 2.0, 3.0),    # Move down
        ],

        "diagonal_sweep": [
            # Takeoff
            (0.0, 0.0, -3.0, 4.0),   # Move up
            # Diagonal patterns
            (4.0, 4.0, 0.0, 10.0),   # Move north-east
            (-4.0, 4.0, 0.0, 10.0),  # Move south-east
            (-4.0, -4.0, 0.0, 10.0), # Move south-west
            (4.0, -4.0, 0.0, 10.0),  # Move north-west
            # Land
            (0.0, 0.0, 3.0, 4.0),    # Move down
        ],

        "spiral_up": [
            # Spiral ascent
            (2.0, 0.0, -1.5, 4.0),   # Move north while ascending
            (2.0, 2.0, -1.5, 4.0),   # Move north-east while ascending
            (0.0, 2.0, -1.5, 4.0),   # Move east while ascending
            (-2.0, 2.0, -1.5, 4.0),  # Move south-east while ascending
            (-2.0, 0.0, -1.5, 4.0),  # Move south while ascending
            # Land
            (0.0, 0.0, 6.0, 6.0),    # Move down
        ],

        "figure_eight": [
            # Takeoff
            (0.0, 0.0, -3.0, 4.0),   # Move up
            # First loop
            (3.0, 2.0, 0.0, 4.0),    # NE
            (2.0, 3.0, 0.0, 4.0),    # E
            (-2.0, 3.0, 0.0, 4.0),   # SE
            (-3.0, 2.0, 0.0, 4.0),   # S
            # Cross center
            (0.0, -2.0, 0.0, 4.0),   # W
            # Second loop
            (3.0, -2.0, 0.0, 4.0),   # NW
            (2.0, -3.0, 0.0, 4.0),   # W
            (-2.0, -3.0, 0.0, 4.0),  # SW
            (-3.0, -2.0, 0.0, 4.0),  # S
            # Land
            (0.0, 0.0, 3.0, 4.0),    # Move down
        ],
    }

    # Add random flight patterns for variety
    # Generate 30 random patterns with different seeds for reproducibility
    NUM_RANDOM_PATTERNS = 30
    for i in range(NUM_RANDOM_PATTERNS):
        pattern_name = f"random_{i:02d}"
        patterns[pattern_name] = generate_random_flight_pattern(
            seed=i,  # Use seed for reproducibility
            num_segments=8,  # 8 random movement segments
            max_velocity=5.0,  # Max 5 m/s horizontal
            max_altitude=15.0  # Max 15m altitude
        )

    # Add some explicitly left-biased patterns for balance
    # (West is negative East direction)
    patterns["west_sweep"] = [
        (0.0, 0.0, -3.0, 4.0),    # Takeoff
        (0.0, -4.0, 0.0, 10.0),   # Move west
        (3.0, -3.0, 0.0, 8.0),    # Move northwest
        (0.0, -4.0, 0.0, 8.0),    # Move west again
        (-3.0, -3.0, 0.0, 8.0),   # Move southwest
        (0.0, 0.0, 3.0, 4.0),     # Land
    ]

    patterns["northwest_arc"] = [
        (0.0, 0.0, -3.0, 4.0),    # Takeoff
        (4.0, -2.0, 0.0, 10.0),   # Move northwest
        (2.0, -4.0, 0.0, 8.0),    # Move more west
        (-2.0, -4.0, 0.0, 8.0),   # Move southwest
        (0.0, 0.0, 3.0, 4.0),     # Land
    ]

    patterns["left_circle"] = [
        (0.0, 0.0, -3.0, 4.0),    # Takeoff
        (-3.0, 0.0, 0.0, 5.0),    # Move south
        (-3.0, -3.0, 0.0, 5.0),   # Move southwest
        (0.0, -3.0, 0.0, 5.0),    # Move west
        (3.0, -3.0, 0.0, 5.0),    # Move northwest
        (3.0, 0.0, 0.0, 5.0),     # Move north
        (0.0, 0.0, 3.0, 4.0),     # Land
    ]

    return patterns


# ============================================================================
# FLIGHT EXECUTION
# ============================================================================

async def execute_flight_pattern(drone, pattern_commands):
    """
    Execute a flight pattern by sending velocity commands to the drone.

    Args:
        drone: Drone object
        pattern_commands: List of (v_north, v_east, v_down, duration) tuples
    """
    for idx, (v_north, v_east, v_down, duration) in enumerate(pattern_commands):
        direction = "up" if v_down < 0 else "down" if v_down > 0 else "level"
        projectairsim_log().info(
            f"  Step {idx+1}/{len(pattern_commands)}: "
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
    """
    Generate complete LiDAR dataset with multiple flight patterns.
    Each flight pattern creates a separate scenario folder.
    """

    # Configuration
    SCENE_FILE = "scene_lidar_drone.jsonc"
    DATASET_ROOT = Path("lidar_dataset_unspoofed")
    ENABLE_VISUALIZATION = False
    MAX_SCENARIOS = 5  # Limit to first 5 flights for testing

    # Create dataset root directory
    DATASET_ROOT.mkdir(exist_ok=True)
    projectairsim_log().info(f"Dataset will be saved to: {DATASET_ROOT.absolute()}")

    # Get all flight patterns
    flight_patterns = get_flight_patterns()
    projectairsim_log().info(f"Will execute {len(flight_patterns)} flight patterns")

    # Initialize client and displays
    client = ProjectAirSimClient()
    image_display = ImageDisplay() if ENABLE_VISUALIZATION else None
    lidar_display = None

    if ENABLE_VISUALIZATION:
        lidar_subwin = image_display.get_subwin_info(2)
        lidar_display = LidarDisplay(
            x=lidar_subwin["x"], y=lidar_subwin["y"] + 30
        )

    try:
        # Connect to simulation
        client.connect()
        projectairsim_log().info("Connected to simulation")

        # Execute each flight pattern (limited to MAX_SCENARIOS)
        flight_patterns_list = list(flight_patterns.items())[:MAX_SCENARIOS]
        for scenario_id, (pattern_name, pattern_commands) in enumerate(flight_patterns_list):
            projectairsim_log().info("")
            projectairsim_log().info("=" * 70)
            projectairsim_log().info(f"SCENARIO {scenario_id+1}/{len(flight_patterns_list)}: {pattern_name}")
            projectairsim_log().info("=" * 70)

            # Create scenario-specific output directory
            scenario_dir = DATASET_ROOT / f"scenario_{scenario_id:04d}_{pattern_name}"
            projectairsim_log().info(f"Output directory: {scenario_dir}")

            # Initialize LiDAR collector for this scenario
            collector = LidarDataCollector(
                output_dir=str(scenario_dir),
                scene_name=SCENE_FILE,
                compute_features=True,
                clear_existing=False  # Keep all scenarios
            )

            # Load scene and create drone
            projectairsim_log().info(f"Loading scene: {SCENE_FILE}")
            world = World(client, SCENE_FILE, delay_after_load_sec=2)
            drone = Drone(client, world, "Drone1")

            # Pass drone reference to collector
            collector.set_drone(drone)

            # Setup visualization if enabled
            if ENABLE_VISUALIZATION:
                # Chase camera
                chase_cam_window = "ChaseCam"
                image_display.add_chase_cam(chase_cam_window)
                client.subscribe(
                    drone.sensors["Chase"]["scene_camera"],
                    lambda _, chase: image_display.receive(chase, chase_cam_window),
                )

                # Down-facing RGB camera
                rgb_name = "RGB-Image"
                image_display.add_image(rgb_name, subwin_idx=0)
                client.subscribe(
                    drone.sensors["DownCamera"]["scene_camera"],
                    lambda _, rgb: image_display.receive(rgb, rgb_name),
                )

                # Depth camera
                depth_name = "Depth-Image"
                image_display.add_image(depth_name, subwin_idx=1)
                client.subscribe(
                    drone.sensors["DownCamera"]["depth_camera"],
                    lambda _, depth: image_display.receive(depth, depth_name),
                )

                image_display.start()

                # LiDAR visualization
                client.subscribe(
                    drone.sensors["lidar1"]["lidar"],
                    lambda _, lidar: lidar_display.receive(lidar),
                )
                lidar_display.start()

            # Subscribe LiDAR data collection
            client.subscribe(
                drone.sensors["lidar1"]["lidar"],
                collector.process_and_store,
            )

            # Enable drone control
            drone.enable_api_control()
            drone.arm()
            projectairsim_log().info("Drone armed and ready")

            # Execute flight pattern
            projectairsim_log().info(f"Executing flight pattern: {pattern_name}")
            await execute_flight_pattern(drone, pattern_commands)

            # Disarm drone
            drone.disarm()
            drone.disable_api_control()
            projectairsim_log().info("Drone disarmed")

            # Save collection summary
            collector.save_summary()

            # Cleanup visualization for next scenario
            if ENABLE_VISUALIZATION:
                image_display.stop()
                lidar_display.stop()

            projectairsim_log().info(f"Scenario {scenario_id+1} complete!")

        # Final summary
        projectairsim_log().info("")
        projectairsim_log().info("=" * 70)
        projectairsim_log().info("DATASET GENERATION COMPLETE")
        projectairsim_log().info("=" * 70)
        projectairsim_log().info(f"Generated {len(flight_patterns_list)} scenarios")
        projectairsim_log().info(f"Dataset location: {DATASET_ROOT.absolute()}")

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect
        client.disconnect()
        if ENABLE_VISUALIZATION:
            if image_display:
                image_display.stop()
            if lidar_display:
                lidar_display.stop()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(generate_dataset())
