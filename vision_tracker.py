"""
Module: vision_tracker.py
Description: High-precision real-time computer vision hand tracking system supporting
             both Bimanual (Dual-Hand) and Single-Hand 6-DOF teleoperation.
Key Features:
  1. Google MediaPipe HandLandmarker with simultaneous dual-hand tracking (num_hands=2).
  2. Dual-Hand Bimanual Mode (Ergonomic decoupled manipulation):
     - Left Hand: Controls base rotation (J1) and shoulder elevation (J2).
     - Right Hand: Controls elbow reach (J3), forearm orientation (J4/J5/J6), and gripper pinch.
  3. Single-Hand Mode:
     - 6-DOF translation and orientation driven concurrently by a single tracked palm.
  4. Seamless automatic transition between Single-Hand and Bimanual teleoperation modes.
  5. Adaptive OneEuroFilter for zero static jitter at rest with instantaneous dynamic tracking.
"""

import os
import sys
import math
import time
import urllib.request
import threading
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
import mediapipe as mp


class LowPassFilter:
    """Standard first-order exponential smoothing low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha: float = float(alpha)
        self.last_value: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = float(alpha)
        if self.last_value is None:
            self.last_value = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self.last_value
        self.last_value = filtered
        return filtered

    def reset(self) -> None:
        self.last_value = None


class OneEuroFilter:
    """
    Adaptive OneEuroFilter for low-latency, jitter-free signal tracking.
    Reference: Casiez et al., CHI 2012 (1-Euro Filter: A Simple Speed-based Low-pass Filter).
    """

    def __init__(self, fc_min: float = 0.8, beta: float = 1.0, d_cutoff: float = 1.0):
        self.fc_min: float = float(fc_min)
        self.beta: float = float(beta)
        self.d_cutoff: float = float(d_cutoff)
        self.x_filter: LowPassFilter = LowPassFilter()
        self.dx_filter: LowPassFilter = LowPassFilter()
        self.last_time: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 1e-6 else 0.01
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(x, 1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 1e-6:
            dt = 0.001
        rate = 1.0 / dt

        prev_x = self.x_filter.last_value
        dx = 0.0 if prev_x is None else (x - prev_x) * rate
        edx = self.dx_filter.filter(dx, self._alpha(rate, self.d_cutoff))
        cutoff = self.fc_min + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(rate, cutoff))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class Point3DOneEuroFilter:
    """3D point filter composing three independent OneEuroFilters along X, Y, and Z."""

    def __init__(self, fc_min: float = 1.0, beta: float = 1.0, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(fc_min, beta, d_cutoff)
        self.fy = OneEuroFilter(fc_min, beta, d_cutoff)
        self.fz = OneEuroFilter(fc_min, beta, d_cutoff)

    def filter(self, point: Tuple[float, float, float], timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        x = self.fx.filter(point[0], timestamp)
        y = self.fy.filter(point[1], timestamp)
        z = self.fz.filter(point[2], timestamp)
        return (x, y, z)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


class CameraStream:
    """Threaded camera capture stream with DirectShow backend for low-latency acquisition."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 60):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.running = False
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            grabbed = self.cap.grab()
            if grabbed:
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                    self.frame_ready.set()
            else:
                time.sleep(0.002)

    def read(self, timeout: float = 0.033) -> Tuple[bool, Optional[np.ndarray]]:
        if self.frame_ready.wait(timeout):
            self.frame_ready.clear()
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()


ThreadedCamera = CameraStream


class HandTracker:
    """Vision tracking engine leveraging MediaPipe HandLandmarker with OneEuroFilter stabilization."""

    def __init__(
        self,
        fc_min: float = 1.0,
        beta: float = 2.0,
        gripper_thresh: float = 0.05,
        homing_dwell_time: float = 0.8,
        homing_radius_px: int = 50,
        deadman_timeout_sec: float = 0.5,
        deadman_max_missing_frames: int = 20,
    ):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("[MediaPipe] Downloading hand_landmarker.task model from Google...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        # Configure model for concurrent bimanual tracking (num_hands=2)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.fc_min = fc_min
        self.beta = beta
        self.gripper_thresh = gripper_thresh
        self.homing_dwell_time = homing_dwell_time
        self.homing_radius_px = homing_radius_px
        self.deadman_timeout_sec = deadman_timeout_sec
        self.deadman_max_missing_frames = deadman_max_missing_frames

        # OneEuroFilters for Left Hand (J1 Base rotation, J2 Shoulder elevation)
        self.filter_left_x = OneEuroFilter(fc_min=fc_min, beta=beta)
        self.filter_left_y = OneEuroFilter(fc_min=fc_min, beta=beta)
        self.filter_left_z = OneEuroFilter(fc_min=fc_min, beta=beta)

        # OneEuroFilters for Right Hand (J3 Elbow pitch, J4 Forearm roll)
        self.filter_right_x = OneEuroFilter(fc_min=fc_min, beta=beta)
        self.filter_right_y = OneEuroFilter(fc_min=fc_min, beta=beta)
        self.filter_right_z = OneEuroFilter(fc_min=fc_min, beta=beta)

        # OneEuroFilters for Right Wrist Orientation (Roll J6, Pitch J5, Yaw J4)
        self.filter_right_roll = OneEuroFilter(fc_min=1.0, beta=1.5)
        self.filter_right_pitch = OneEuroFilter(fc_min=1.0, beta=1.5)
        self.filter_right_yaw = OneEuroFilter(fc_min=1.0, beta=1.5)

        # Backward compatibility aliases for Single-Hand mode
        self.filter_x = self.filter_right_x
        self.filter_y = self.filter_right_y
        self.filter_z = self.filter_right_z
        self.filter_roll = self.filter_right_roll
        self.filter_pitch = self.filter_right_pitch
        self.filter_yaw = self.filter_right_yaw

        self.gripper_state: int = 0
        self.control_mode: str = "SINGLE_HAND"

        self.is_homed: bool = False
        self.origin_pos: Optional[Tuple[float, float, float]] = None
        self.origin_left: Optional[Tuple[float, float, float]] = None
        self.origin_right: Optional[Tuple[float, float, float]] = None
        self.dual_origin_left: Optional[Tuple[float, float, float]] = None
        self.dual_origin_right: Optional[Tuple[float, float, float]] = None
        self.single_origin: Optional[Tuple[float, float, float]] = None
        self.prev_mode: Optional[str] = None
        self.homing_progress: float = 1.0

        self.last_detection_time: float = time.perf_counter()
        self.missing_frame_count: int = 0
        self.deadman_active: bool = False

        self.last_valid_pos: Optional[Tuple[float, float, float]] = None
        self.last_valid_left_pos: Optional[Tuple[float, float, float]] = None
        self.last_valid_right_pos: Optional[Tuple[float, float, float]] = None
        self.last_valid_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_right_lms: Optional[List[Tuple[float, float, float]]] = None
        self.current_left_lms: Optional[List[Tuple[float, float, float]]] = None

        self._start_time = time.perf_counter()
        self._last_timestamp_ms: int = -1

    def reset_homing(self) -> None:
        """Reset origin benchmarks and reinitialize all smoothing filters."""
        self.is_homed = False
        self.origin_pos = None
        self.origin_left = None
        self.origin_right = None
        self.dual_origin_left = None
        self.dual_origin_right = None
        self.single_origin = None
        self.prev_mode = None
        self.filter_left_x.reset()
        self.filter_left_y.reset()
        self.filter_left_z.reset()
        self.filter_right_x.reset()
        self.filter_right_y.reset()
        self.filter_right_z.reset()
        self.filter_right_roll.reset()
        self.filter_right_pitch.reset()
        self.filter_right_yaw.reset()

    @staticmethod
    def _compute_palm_pos_size(lms: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        wrist = lms[0]
        index_mcp = lms[5]
        middle_mcp = lms[9]
        pinky_mcp = lms[17]

        palm_cx = (wrist[0] + index_mcp[0] + middle_mcp[0] + pinky_mcp[0]) * 0.25
        palm_cy = (wrist[1] + index_mcp[1] + middle_mcp[1] + pinky_mcp[1]) * 0.25

        span_v = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
        span_h = math.hypot(pinky_mcp[0] - index_mcp[0], pinky_mcp[1] - index_mcp[1])
        palm_size = (span_v + span_h) * 0.5
        return palm_cx, palm_cy, palm_size

    @staticmethod
    def _compute_hand_angles(lms: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        wrist = lms[0]
        index_mcp = lms[5]
        middle_mcp = lms[9]
        pinky_mcp = lms[17]

        ux = middle_mcp[0] - wrist[0]
        uy = middle_mcp[1] - wrist[1]
        uz = middle_mcp[2] - wrist[2]

        vx = index_mcp[0] - pinky_mcp[0]
        vy = index_mcp[1] - pinky_mcp[1]
        vz = index_mcp[2] - pinky_mcp[2]

        raw_roll = math.degrees(math.atan2(ux, -uy)) * 1.3
        raw_pitch = -math.degrees(math.atan2(uz, max(1e-5, math.hypot(ux, uy)))) * 2.2
        raw_yaw = math.degrees(math.atan2(vz, max(1e-5, math.hypot(vx, vy)))) * 1.5

        raw_roll = max(-85.0, min(85.0, raw_roll))
        raw_pitch = max(-85.0, min(85.0, raw_pitch))
        raw_yaw = max(-75.0, min(75.0, raw_yaw))
        return raw_roll, raw_pitch, raw_yaw

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Process video frame and extract filtered landmark telemetry."""
        now = time.perf_counter()
        raw_ts = int((now - self._start_time) * 1000)
        if raw_ts <= self._last_timestamp_ms:
            raw_ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = raw_ts
        timestamp_ms = raw_ts
        h, w, _ = frame_bgr.shape

        mirrored_bgr = cv2.flip(frame_bgr, 1)
        rgb_frame = cv2.cvtColor(mirrored_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detect_res = self.detector.detect_for_video(mp_image, timestamp_ms)

        detected_hands = []
        if detect_res.hand_landmarks and len(detect_res.hand_landmarks) > 0:
            for lms in detect_res.hand_landmarks:
                pts = [(lm.x, lm.y, lm.z) for lm in lms]
                avg_x = sum(p[0] for p in pts) / len(pts)
                detected_hands.append({"landmarks": pts, "avg_x": avg_x})

        # Sort detected hands horizontally across mirrored display (x_min = left, x_max = right)
        detected_hands.sort(key=lambda item: item["avg_x"])

        hand_detected = False
        pos_hand_lms = None
        right_lms = None
        left_lms = None
        cur_left_pos = None
        cur_right_pos = None
        single_pos = None

        roll_deg, pitch_deg, yaw_deg = self.last_valid_angles
        pinch_dist = 1.0

        if len(detected_hands) >= 2:
            # === DUAL-HAND BIMANUAL JOINT CONTROL MODE ===
            hand_detected = True
            self.control_mode = "DUAL_HAND"
            left_lms = detected_hands[0]["landmarks"]   # Left screen landmark cluster
            right_lms = detected_hands[1]["landmarks"]  # Right screen landmark cluster

            self.current_right_lms = right_lms
            self.current_left_lms = left_lms
            self.missing_frame_count = 0
            self.last_detection_time = now

            # 1. Filter Left Hand (J1 Base rotation & J2 Shoulder elevation)
            l_cx, l_cy, l_size = self._compute_palm_pos_size(left_lms)
            fl_x = self.filter_left_x.filter(l_cx, now)
            fl_y = self.filter_left_y.filter(l_cy, now)
            fl_z = self.filter_left_z.filter(l_size, now)
            cur_left_pos = (fl_x, fl_y, fl_z)
            self.last_valid_left_pos = cur_left_pos

            # 2. Filter Right Hand (J3 Elbow pitch & J4 Forearm roll)
            r_cx, r_cy, r_size = self._compute_palm_pos_size(right_lms)
            fr_x = self.filter_right_x.filter(r_cx, now)
            fr_y = self.filter_right_y.filter(r_cy, now)
            fr_z = self.filter_right_z.filter(r_size, now)
            cur_right_pos = (fr_x, fr_y, fr_z)
            self.last_valid_right_pos = cur_right_pos
            self.last_valid_pos = cur_right_pos

            # Automatic origin benchmark initialization
            if not self.is_homed:
                self.is_homed = True
            if self.dual_origin_left is None:
                self.dual_origin_left = cur_left_pos
            if self.dual_origin_right is None:
                self.dual_origin_right = cur_right_pos
            if self.single_origin is None:
                self.single_origin = cur_right_pos

            self.origin_left = self.dual_origin_left
            self.origin_right = self.dual_origin_right
            self.origin_pos = self.dual_origin_right
            self.prev_mode = "DUAL_HAND"

            # 3. Compute Right Wrist Orientation (Roll J6, Pitch J5, Yaw J4)
            raw_roll, raw_pitch, raw_yaw = self._compute_hand_angles(right_lms)
            s_roll = self.filter_right_roll.filter(raw_roll, now)
            s_pitch = self.filter_right_pitch.filter(raw_pitch, now)
            s_yaw = self.filter_right_yaw.filter(raw_yaw, now)
            roll_deg, pitch_deg, yaw_deg = (s_roll, s_pitch, s_yaw)
            self.last_valid_angles = (roll_deg, pitch_deg, yaw_deg)

            # 4. Gripper Pinch Detection (Thumb tip + Index tip of right hand)
            thumb_tip = right_lms[4]
            index_tip = right_lms[8]
            raw_pinch = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
            rel_pinch = raw_pinch / max(0.01, r_size)
            pinch_dist = raw_pinch
            if self.gripper_state == 0:
                if rel_pinch < 0.38 or raw_pinch < 0.055:
                    self.gripper_state = 1
            else:
                if rel_pinch > 0.55 and raw_pinch > 0.075:
                    self.gripper_state = 0

            pos_hand_lms = right_lms

        elif len(detected_hands) == 1:
            # === SINGLE-HAND CONTROL MODE ===
            hand_detected = True
            self.control_mode = "SINGLE_HAND"
            single_lms = detected_hands[0]["landmarks"]
            s_cx, s_cy, s_size = self._compute_palm_pos_size(single_lms)

            # Filter single hand Cartesian coordinates
            fs_x = self.filter_x.filter(s_cx, now)
            fs_y = self.filter_y.filter(s_cy, now)
            fs_z = self.filter_z.filter(s_size, now)
            single_pos = (fs_x, fs_y, fs_z)
            self.last_valid_pos = single_pos
            cur_right_pos = single_pos

            # Initialize benchmark origin once
            if not self.is_homed:
                self.is_homed = True
            if self.single_origin is None:
                self.single_origin = single_pos
            if self.dual_origin_right is None:
                self.dual_origin_right = single_pos

            self.origin_pos = self.single_origin
            self.origin_right = self.single_origin
            self.prev_mode = "SINGLE_HAND"

            raw_roll, raw_pitch, raw_yaw = self._compute_hand_angles(single_lms)
            s_roll = self.filter_right_roll.filter(raw_roll, now)
            s_pitch = self.filter_right_pitch.filter(raw_pitch, now)
            s_yaw = self.filter_right_yaw.filter(raw_yaw, now)
            roll_deg, pitch_deg, yaw_deg = (s_roll, s_pitch, s_yaw)
            self.last_valid_angles = (roll_deg, pitch_deg, yaw_deg)

            thumb_tip = single_lms[4]
            index_tip = single_lms[8]
            raw_pinch = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
            rel_pinch = raw_pinch / max(0.01, s_size)
            pinch_dist = raw_pinch
            if self.gripper_state == 0:
                if rel_pinch < 0.38 or raw_pinch < 0.055:
                    self.gripper_state = 1
            else:
                if rel_pinch > 0.55 and raw_pinch > 0.075:
                    self.gripper_state = 0

            self.current_right_lms = single_lms
            self.current_left_lms = None
            self.missing_frame_count = 0
            self.last_detection_time = now
            pos_hand_lms = single_lms
            right_lms = single_lms

        else:
            # Temporary tracking loss: persist prior landmarks for up to 20 frames
            self.missing_frame_count += 1
            if self.missing_frame_count < 20 and (self.current_right_lms or self.current_left_lms):
                hand_detected = True
                right_lms = self.current_right_lms
                left_lms = self.current_left_lms
                pos_hand_lms = self.current_right_lms or self.current_left_lms

        dt_lost = now - self.last_detection_time
        self.deadman_active = (dt_lost > self.deadman_timeout_sec) or (self.missing_frame_count >= self.deadman_max_missing_frames)

        return {
            "mirrored_frame": mirrored_bgr,
            "hand_detected": hand_detected,
            "control_mode": self.control_mode,
            "num_hands_detected": len(detected_hands),
            "right_hand": right_lms,
            "left_hand": left_lms,
            "landmarks": pos_hand_lms,
            "filtered_pos": self.last_valid_pos,
            "origin_pos": self.single_origin or self.origin_pos,
            "single_pos": single_pos,
            "single_origin": self.single_origin,
            "current_left_pos": cur_left_pos,
            "current_right_pos": cur_right_pos,
            "left_pos": cur_left_pos,
            "right_pos": cur_right_pos,
            "origin_left": self.dual_origin_left or self.origin_left,
            "origin_right": self.dual_origin_right or self.origin_right,
            "right_angles": (roll_deg, pitch_deg, yaw_deg),
            "is_homed": self.is_homed,
            "homing_progress": self.homing_progress,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pinch_distance": pinch_dist,
            "gripper_state": self.gripper_state,
            "deadman_active": self.deadman_active,
            "frame_width": w,
            "frame_height": h,
        }

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        track_data: Optional[Dict[str, Any]] = None,
        w: int = 640,
        h: int = 480,
        **kwargs
    ) -> None:
        """Render anatomical skeleton overlay and visual status markers."""
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky finger
            (5, 9), (9, 13), (13, 17)               # Palm base
        ]

        if track_data is None:
            track_data = {}
        for k, v in kwargs.items():
            if k not in track_data:
                track_data[k] = v

        mode = track_data.get("control_mode", "SINGLE_HAND")
        right_lms = track_data.get("right_hand") or track_data.get("landmarks")
        left_lms = track_data.get("left_hand")
        grip = track_data.get("gripper_state", 0)

        def draw_single_hand_mesh(lms, bone_color, joint_color, label_text, label_color):
            for p1_idx, p2_idx in CONNECTIONS:
                pt1 = (int(lms[p1_idx][0] * w), int(lms[p1_idx][1] * h))
                pt2 = (int(lms[p2_idx][0] * w), int(lms[p2_idx][1] * h))
                cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            for idx, lm in enumerate(lms):
                px = (int(lm[0] * w), int(lm[1] * h))
                if idx in [4, 8]:
                    c = (40, 40, 240) if grip == 1 else (0, 255, 255)
                    cv2.circle(frame, px, 6, c, -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, px, 3, joint_color, -1, cv2.LINE_AA)

            wrist = lms[0]
            middle_mcp = lms[9]
            cx = int((wrist[0] + middle_mcp[0]) * 0.5 * w)
            cy = int((wrist[1] + middle_mcp[1]) * 0.5 * h)
            cv2.circle(frame, (cx, cy), 10, label_color, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (cx, cy), label_color, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

        if mode == "DUAL_HAND":
            # Render Left Hand (J1 Base rotation + J2 Shoulder elevation) in Gold/Amber
            if left_lms:
                draw_single_hand_mesh(
                    left_lms,
                    bone_color=(0, 200, 255),
                    joint_color=(100, 255, 255),
                    label_text="LEFT HAND: J1 (Base) & J2 (Shoulder)",
                    label_color=(0, 215, 255),
                )
            # Render Right Hand (J3 Elbow + J4 Forearm roll + Gripper) in Cyan / Red
            if right_lms:
                pos_color = (40, 40, 240) if grip == 1 else (255, 230, 0)
                draw_single_hand_mesh(
                    right_lms,
                    bone_color=pos_color,
                    joint_color=(255, 255, 255),
                    label_text="RIGHT HAND: J3 (Elbow) & J4 (Forearm) + Grip",
                    label_color=(255, 230, 0),
                )
        else:
            # Render Single Hand in Cyan
            if right_lms:
                draw_single_hand_mesh(
                    right_lms,
                    bone_color=(255, 230, 0),
                    joint_color=(255, 255, 255),
                    label_text="SINGLE HAND: J1-J6 & GRIP",
                    label_color=(255, 230, 0),
                )

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()


VisionTracker = HandTracker
