#!/usr/bin/python3
# coding=utf8

"""
Demo for "added functionality" (Sorting perception):
- Read frames from the ArmPi camera
- Run BlockPerception in mode="sorting"
- Shows:
    * bounding box + world coords
    * "Color: <name>" label (voted over 3 frames)
Press ESC to quit.
"""

import sys
sys.path.append('/home/pi/ArmPi/')  # import ArmPi helpers/config

import cv2
import Camera

from LABConfig import color_range
from CameraCalibration.CalibrationConfig import square_length
from ArmIK.Transform import getCenter, convertCoordinate, getROI, getMaskROI

from block_perception import BlockPerception


def main():
    range_rgb = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'None': (0, 0, 0),
    }

    # Same as vendor code uses in multiple functions
    size = (640, 480)

    perceiver = BlockPerception(
        color_range=color_range,
        range_rgb=range_rgb,
        size=size,
        square_length=square_length,
        getCenter_fn=getCenter,
        convertCoordinate_fn=convertCoordinate,
        getROI_fn=getROI,
        getMaskROI_fn=getMaskROI,
        # Sorting-like defaults (match ColorSorting.py behavior)
        sorting_stable_dist=0.5,
        sorting_stable_time=1.0,
        vote_window=3,
    )

    cam = Camera.Camera()
    cam.camera_open()

    # For sorting perception, it's typical to consider multiple colors
    target_colors = ('red', 'green', 'blue')

    while True:
        frame = cam.frame
        if frame is None:
            continue

        img = frame.copy()

        annotated, det = perceiver.process_frame(
            img_bgr=img,
            target_colors=target_colors,
            is_running=True,
            action_finish=True,   # demo only; arm is not moving
            mode="sorting",
        )

        # Print when it decides "stable enough"
        if det.get("start_pick_up", False) and det.get("stable_world_xy") is not None:
            wx, wy = det["stable_world_xy"]
            print("Stable:", det.get("detect_color"), "world:", (wx, wy), "angle:", det.get("rotation_angle"))

        cv2.imshow("Sorting Perception Demo (ESC to quit)", annotated)
        if cv2.waitKey(1) == 27:
            break

    cam.camera_close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
