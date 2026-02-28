"""
Copyright (C) Microsoft Corporation. 
Copyright (C) 2025 IAMAI CONSULTING CORP
MIT License.

Demonstrates flying a quadrotor drone with camera sensors.
"""

import asyncio

import cv2
import numpy as np

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.image_utils import ImageDisplay

# Async main function to wrap async drone commands
async def main():
    # Create a Project AirSim client
    client = ProjectAirSimClient()

    # Initialize an ImageDisplay object to display camera sub-windows
    image_display = ImageDisplay()

    # VideoWriter for uncompressed RGB recording (DIB = Device Independent Bitmap, uncompressed)
    # Will be initialized on the first frame so we can match the source resolution.
    video_writer: cv2.VideoWriter | None = None
    video_fps = 30.0
    video_output_path = "rgb_recording.avi"

    def write_frame(image):
        nonlocal video_writer
        if image is None or "data" not in image or len(image["data"]) == 0:
            return
        frame = unpack_image(image)  # -> BGR numpy array
        if frame is None:
            return
        h, w = frame.shape[:2]
        if video_writer is None:
            # fourcc 0 = uncompressed raw; 'DIB ' is the Windows uncompressed codec
            fourcc = cv2.VideoWriter_fourcc(*"DIB ")
            video_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (w, h))
            projectairsim_log().info(f"Recording uncompressed video to {video_output_path} ({w}x{h} @{video_fps}fps)")
        if len(frame.shape) == 2:                      # grayscale -> convert to BGR for VideoWriter
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)

    try:
        # Connect to simulation environment
        client.connect()

        # Create a World object to interact with the sim world and load a scene
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)

        # Create a Drone object to interact with a drone in the loaded sim world
        drone = Drone(client, world, "Drone1")

        # ------------------------------------------------------------------------------

        # Subscribe to chase camera sensor as a client-side pop-up window
        chase_cam_window = "ChaseCam"
        image_display.add_chase_cam(chase_cam_window)
        client.subscribe(
            drone.sensors["Chase"]["scene_camera"],
            lambda _, chase: image_display.receive(chase, chase_cam_window),
        )

        # Subscribe to the downward-facing camera sensor's RGB and Depth images
        rgb_name = "RGB-Image"
        image_display.add_image(rgb_name, subwin_idx=0)
        client.subscribe(
            drone.sensors["DownCamera"]["scene_camera"],
            lambda _, rgb: [image_display.receive(rgb, rgb_name), write_frame(rgb)],
        )

        depth_name = "Depth-Image"
        image_display.add_image(depth_name, subwin_idx=2)
        client.subscribe(
            drone.sensors["DownCamera"]["depth_camera"],
            lambda _, depth: image_display.receive(depth, depth_name),
        )

        image_display.start()

        # ------------------------------------------------------------------------------

        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()

        # ------------------------------------------------------------------------------

        projectairsim_log().info("takeoff_async: starting")
        takeoff_task = (
            await drone.takeoff_async()
        )  # schedule an async task to start the command

        # Example 1: Wait on the result of async operation using 'await' keyword
        await takeoff_task
        projectairsim_log().info("takeoff_async: completed")

        # ------------------------------------------------------------------------------

        # Command the drone to move up in NED coordinate system at 1 m/s for 4 seconds
        move_up_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-1.0, duration=4.0
        )
        projectairsim_log().info("Move-Up invoked")

        await move_up_task
        projectairsim_log().info("Move-Up completed")

        # ------------------------------------------------------------------------------

        # Command the Drone to move down in NED coordinate system at 1 m/s for 4 seconds
        move_down_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=1.0, duration=4.0
        )  # schedule an async task to start the command
        projectairsim_log().info("Move-Down invoked")

        # Example 2: Wait for move_down_task to complete before continuing
        while not move_down_task.done():
            await asyncio.sleep(0.005)
        projectairsim_log().info("Move-Down completed")

        # ------------------------------------------------------------------------------

        projectairsim_log().info("land_async: starting")
        land_task = await drone.land_async()
        await land_task
        projectairsim_log().info("land_async: completed")

        # ------------------------------------------------------------------------------

        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()

        # ------------------------------------------------------------------------------

    # logs exception on the console
    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()

        image_display.stop()

        if video_writer is not None:
            video_writer.release()
            projectairsim_log().info(f"Video saved to {video_output_path}")


if __name__ == "__main__":
    asyncio.run(main())  # Runner for async main function
