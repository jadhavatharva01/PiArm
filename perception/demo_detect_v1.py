#!/usr/bin/python3
# coding=utf8

"""
Demo for Task 5:
- Read frames from the ArmPi camera
- Run refactored perception (BlockPerception)
- Draw box + (world_x, world_y) on the display
Press ESC to quit.
"""

import sys
sys.path.append('/home/pi/ArmPi/')  # so we can import ArmPi calibration + helpers

import cv2
import Camera

from LABConfig import color_range
from CameraCalibration.CalibrationConfig import square_length
from ArmIK.Transform import getCenter, convertCoordinate, getROI, getMaskROI

from block_perception import BlockPerception

size = (640, 480)

def main():
    # Same RGB mapping used in ColorTracking.py for debug drawing
    range_rgb = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
    }

    # Create perception object
    perceiver = BlockPerception(
        color_range=color_range,
        range_rgb=range_rgb,
        size=size,
        square_length=square_length,
        getCenter_fn=getCenter,
        convertCoordinate_fn=convertCoordinate,
        getROI_fn=getROI,
        getMaskROI_fn=getMaskROI,
    )

    # Camera
    cam = Camera.Camera()
    cam.camera_open()

    # Choose which colors to detect
    target_colors = ('red',)  # change to ('red','green','blue') if needed

    while True:
        frame = cam.frame
        if frame is None:
            continue

        img = frame.copy()

        annotated, det = perceiver.process_frame(
            img_bgr=img,
            target_colors=target_colors,
            is_running=True,
            action_finish=True,  # in this demo, we never move the arm
        )

        # Optional: print stable coordinate when ready-to-pick triggers
        if det.get("start_pick_up", False) and det.get("stable_world_xy") is not None:
            wx, wy = det["stable_world_xy"]
            print("Stable target:", det.get("detect_color"), "world:", (wx, wy), "angle:", det.get("rotation_angle"))

        cv2.imshow("Perception Demo (ESC to quit)", annotated)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.camera_close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
