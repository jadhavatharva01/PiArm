#!/usr/bin/python3
# coding=utf8

"""
Palletizing perception demo:
Same as sorting demo, but uses mode="palletizing" (stability time ~0.5s).
Press ESC to quit.
"""

import sys
sys.path.append('/home/pi/ArmPi/')

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
        palletizing_stable_dist=0.5,
        palletizing_stable_time=0.5,
        vote_window=3,
    )

    cam = Camera.Camera()
    cam.camera_open()

    target_colors = ('red', 'green', 'blue')

    while True:
        frame = cam.frame
        if frame is None:
            continue

        annotated, det = perceiver.process_frame(
            img_bgr=frame.copy(),
            target_colors=target_colors,
            is_running=True,
            action_finish=True,
            mode="palletizing",
        )

        if det.get("start_pick_up", False) and det.get("stable_world_xy") is not None:
            wx, wy = det["stable_world_xy"]
            print("Palletizing stable:", det.get("detect_color"), "world:", (wx, wy), "angle:", det.get("rotation_angle"))

        cv2.imshow("Palletizing Perception Demo (ESC to quit)", annotated)
        if cv2.waitKey(1) == 27:
            break

    cam.camera_close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
