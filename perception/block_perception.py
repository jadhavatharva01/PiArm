#!/usr/bin/python3
# coding=utf8

import time
import math
import numpy as np
import cv2


class BlockPerception:
    """
    Refactored perception module extracted from ColorTracking.run(img),
    extended to support ColorSorting-style functionality.

    Input:  BGR frame (OpenCV)
    Output: (annotated_frame, detection_dict)

    Modes:
      - mode="tracking": behavior similar to ColorTracking.py
      - mode="sorting": behavior similar to ColorSorting.py
          * ROI applied when get_roi and NOT start_pick_up
          * color voting over vote_window frames
          * different stability thresholds (distance/time)
          * overlays "Color: <name>"

    detection_dict keys (when detected):
      - detect_color
      - area_max
      - rect
      - box
      - roi
      - img_center (cx, cy)
      - world_xy (world_x, world_y) instantaneous
      - stable_world_xy (world_X, world_Y) when stable enough
      - rotation_angle
      - start_pick_up (bool)
      - track (bool)
      - distance
    """

    def __init__(
        self,
        color_range,
        range_rgb,
        size,
        square_length,
        getCenter_fn,
        convertCoordinate_fn,
        getROI_fn,
        getMaskROI_fn,
        kernel_size=(6, 6),
        blur_ksize=(11, 11),
        blur_sigma=11,
        min_area=2500,
        # tracking defaults (ColorTracking-like)
        stable_dist=0.3,
        stable_time=1.5,
        # sorting defaults (ColorSorting-like)
        sorting_stable_dist=0.5,
        sorting_stable_time=1.0,
        # sorting vote window
        vote_window=3,
        palletizing_stable_dist=0.5,
        palletizing_stable_time=0.5,
    ):
        # External configs / functions from your codebase
        self.color_range = color_range
        self.range_rgb = range_rgb
        self.size = size
        self.square_length = square_length

        self.getCenter = getCenter_fn
        self.convertCoordinate = convertCoordinate_fn
        self.getROI = getROI_fn
        self.getMaskROI = getMaskROI_fn

        # Tunable perception knobs
        self.kernel = np.ones(kernel_size, np.uint8)
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.min_area = min_area

        # thresholds
        self.stable_dist = stable_dist
        self.stable_time = stable_time
        self.sorting_stable_dist = sorting_stable_dist
        self.sorting_stable_time = sorting_stable_time
        self.palletizing_stable_dist = palletizing_stable_dist
        self.palletizing_stable_time = palletizing_stable_time

        # Sorting-mode: vote color over N frames to reduce flicker
        self.color_votes = []
        self.vote_window = vote_window

        # Stateful tracking variables (same intent as original globals)
        self.roi = ()
        self.get_roi = False
        self.last_x, self.last_y = 0.0, 0.0

        self.center_list = []
        self.count = 0
        self.start_count_t1 = True
        self.t1 = 0.0

        self.track = False
        self.start_pick_up = False
        self.rotation_angle = 0.0

    # ---------- helper methods (match your flowchart boxes) ----------

    def preprocess(self, img_bgr):
        img_resized = cv2.resize(img_bgr, self.size, interpolation=cv2.INTER_NEAREST)
        img_blur = cv2.GaussianBlur(img_resized, self.blur_ksize, self.blur_sigma)
        return img_blur

    def maybe_apply_roi_mask(self, frame_bgr, mode, start_pick_up):
        """
        Matches original code differences:
          - tracking: apply ROI when get_roi and start_pick_up
          - sorting:  apply ROI when get_roi and NOT start_pick_up
        """
        if not self.get_roi:
            return frame_bgr

        if mode == "sorting":
            if not start_pick_up:
                self.get_roi = False
                return self.getMaskROI(frame_bgr, self.roi, self.size)
            return frame_bgr

        # default: tracking behavior
        if start_pick_up:
            self.get_roi = False
            return self.getMaskROI(frame_bgr, self.roi, self.size)
        return frame_bgr

    def to_lab(self, frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    def segment_color(self, frame_lab, detect_color):
        lo, hi = self.color_range[detect_color]
        return cv2.inRange(frame_lab, lo, hi)

    def clean_mask(self, mask):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        return closed

    def find_contours(self, mask):
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    @staticmethod
    def get_area_max_contour(contours):
        contour_area_max = 0.0
        area_max_contour = None
        for c in contours:
            area = abs(cv2.contourArea(c))
            if area > contour_area_max:
                contour_area_max = area
                # original had extra noise filter at 300
                if area > 300:
                    area_max_contour = c
        return area_max_contour, contour_area_max

    def select_largest_target(self, frame_lab, target_colors):
        """
        ColorSorting-style behavior: truly pick max-area across all target colors.
        """
        best = {
            "detect_color": None,
            "area_max": 0.0,
            "areaMaxContour": None,
        }

        for c in self.color_range.keys():
            if c in target_colors:
                mask = self.segment_color(frame_lab, c)
                mask = self.clean_mask(mask)
                contours = self.find_contours(mask)
                areaMaxContour, area_max = self.get_area_max_contour(contours)
                if areaMaxContour is not None and area_max > best["area_max"]:
                    best.update(
                        detect_color=c,
                        area_max=area_max,
                        areaMaxContour=areaMaxContour,
                    )
        return best

    def estimate_pose_and_coords(self, areaMaxContour):
        rect = cv2.minAreaRect(areaMaxContour)
        box = np.int0(cv2.boxPoints(rect))

        roi = self.getROI(box)
        self.roi = roi
        self.get_roi = True

        img_centerx, img_centery = self.getCenter(rect, roi, self.size, self.square_length)
        world_x, world_y = self.convertCoordinate(img_centerx, img_centery, self.size)

        return rect, box, roi, (img_centerx, img_centery), (world_x, world_y)

    def vote_color(self, color_name):
        """
        Sorting-style color stabilization:
        Collect N votes, then return a stabilized color.
        Returns stable_color_name or None if not enough votes yet.
        """
        mapping = {"red": 1, "green": 2, "blue": 3}
        v = mapping.get(color_name, 0)
        self.color_votes.append(v)

        if len(self.color_votes) < self.vote_window:
            return None

        avg = float(np.mean(np.array(self.color_votes)))
        self.color_votes = []

        color_id = int(round(avg))
        inv = {1: "red", 2: "green", 3: "blue"}
        return inv.get(color_id, "None")

    def stability_update(self, world_x, world_y, action_finish=True, mode="tracking"):
        """
        Mirrors distance+time gating (tracking) and uses sorting thresholds if mode="sorting".
        Returns: (distance, stable_world_xy_or_None, start_pick_up_flag)
        """
        if mode == "sorting":
            dist_th = self.sorting_stable_dist
            time_th = self.sorting_stable_time
        elif mode == "palletizing":
            dist_th = self.palletizing_stable_dist
            time_th = self.palletizing_stable_time
        else:
            dist_th = self.stable_dist
            time_th = self.stable_time

        distance = math.sqrt((world_x - self.last_x) ** 2 + (world_y - self.last_y) ** 2)
        self.last_x, self.last_y = world_x, world_y

        stable_world = None
        start_pick_up = False

        if action_finish:
            if distance < dist_th:
                self.center_list.extend((world_x, world_y))
                self.count += 1

                if self.start_count_t1:
                    self.start_count_t1 = False
                    self.t1 = time.time()

                if time.time() - self.t1 > time_th:
                    stable_world = tuple(np.mean(np.array(self.center_list).reshape(self.count, 2), axis=0))
                    self.center_list = []
                    self.count = 0
                    self.start_count_t1 = True
                    start_pick_up = True
            else:
                self.t1 = time.time()
                self.start_count_t1 = True
                self.center_list = []
                self.count = 0

        return distance, stable_world, start_pick_up

    def annotate(self, img, box, detect_color, world_x, world_y):
        color = self.range_rgb.get(detect_color, (0, 0, 0))
        cv2.drawContours(img, [box], -1, color, 2)
        cv2.putText(
            img,
            f"({world_x},{world_y})",
            (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        return img

    def annotate_color_label(self, img, detect_color):
        draw_color = self.range_rgb.get(detect_color, (0, 0, 0))
        cv2.putText(
            img,
            "Color: " + str(detect_color),
            (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            draw_color,
            2,
        )
        return img

    # ---------- main API ----------

    def process_frame(self, img_bgr, target_colors, is_running=True, action_finish=True, mode="tracking"):
        """
        Equivalent role to ColorTracking.run(img), but returns structured detection.
        Set mode="sorting" to mimic ColorSorting.py perception extras.
        """
        annotated = img_bgr
        h, w = annotated.shape[:2]
        cv2.line(annotated, (0, h // 2), (w, h // 2), (0, 0, 200), 1)
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 0, 200), 1)

        if not is_running:
            return annotated, {"track": False, "start_pick_up": False}

        frame = self.preprocess(img_bgr)
        frame = self.maybe_apply_roi_mask(frame, mode=mode, start_pick_up=self.start_pick_up)
        lab = self.to_lab(frame)

        # detect only when not in pickup phase (mirrors original)
        if self.start_pick_up:
            det = {"track": False, "start_pick_up": True}
            if mode == "sorting":
                annotated = self.annotate_color_label(annotated, "None")
            return annotated, det

        best = self.select_largest_target(lab, target_colors)

        if best["area_max"] <= self.min_area or best["areaMaxContour"] is None:
            self.track = False
            det = {"track": False, "start_pick_up": False, "detect_color": "None"}
            if mode == "sorting":
                annotated = self.annotate_color_label(annotated, "None")
            return annotated, det

        # Sorting mode: vote/stabilize the detected color over a short window
        if mode == "sorting":
            voted = self.vote_color(best["detect_color"])
            if voted is not None:
                best["detect_color"] = voted

        rect, box, roi, img_center, world_xy = self.estimate_pose_and_coords(best["areaMaxContour"])
        world_x, world_y = world_xy

        annotated = self.annotate(annotated, box, best["detect_color"], world_x, world_y)

        self.track = True
        distance, stable_world, start_pick = self.stability_update(
            world_x, world_y, action_finish=action_finish, mode=mode
        )

        if start_pick:
            self.rotation_angle = rect[2]
            self.start_pick_up = True

        det = {
            "track": True,
            "detect_color": best["detect_color"],
            "area_max": best["area_max"],
            "rect": rect,
            "box": box,
            "roi": roi,
            "img_center": img_center,
            "world_xy": (world_x, world_y),
            "distance": distance,
            "stable_world_xy": stable_world,
            "rotation_angle": self.rotation_angle,
            "start_pick_up": self.start_pick_up,
        }

        if mode == "sorting":
            annotated = self.annotate_color_label(annotated, det.get("detect_color", "None"))

        return annotated, det
