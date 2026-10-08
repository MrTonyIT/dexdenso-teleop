"""
Module: main.py
Description: Main orchestrator and cyber telemetry HUD for the DexDenso Teleoperation Framework.
Integrates:
  1. ThreadedCamera: Zero-latency asynchronous frame acquisition backend.
  2. HandTracker: Dual-hand MediaPipe tracking, adaptive OneEuroFilter, gripper pinch, and deadman safety.
  3. KinematicsMapper: 6-DOF analytical inverse kinematics with industrial reach envelopes.
  4. WincapsBridge: 60Hz TCP client for DENSO RC8 and dual UDP broadcast for NVIDIA Isaac Sim.
  5. Cyber Telemetry HUD: Real-time pose overlay and joint telemetry feedback.
"""

import sys
import time
import math
from typing import Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from vision_tracker import ThreadedCamera, HandTracker
from kinematics_mapper import KinematicsMapper
from wincaps_bridge import WincapsBridge


class TeleopApp:
    """Main teleoperation application orchestrating vision tracking, kinematics, and networking."""

    # Cyber/Robotics HUD Color Palette (BGR format)
    COLOR_BG_CARD = (20, 24, 28)
    COLOR_CYAN = (255, 230, 0)       # Neon cyan
    COLOR_GREEN = (50, 220, 50)      # Operational green
    COLOR_YELLOW = (0, 215, 255)     # Warning amber
    COLOR_RED = (40, 40, 240)        # Alarm / Gripper closed
    COLOR_WHITE = (240, 240, 240)
    COLOR_GRAY = (130, 130, 130)
    COLOR_ORANGE = (0, 140, 255)

    def __init__(
        self,
        camera_id: int = 0,
        rc8_host: str = "127.0.0.1",
        rc8_port: int = 5000,
    ):
        print("=" * 70)
        print("  DEXDENSO: ZERO-LATENCY VISION TELEOPERATION FRAMEWORK")
        print("=" * 70)

        # 1. Initialize threaded camera stream
        print("[1/4] Initializing low-latency camera acquisition (ThreadedCamera)...")
        self.camera = ThreadedCamera(camera_id=camera_id, width=640, height=480, fps=60).start()

        # 2. Initialize vision tracking and adaptive OneEuroFilter
        print("[2/4] Loading MediaPipe HandLandmarker & OneEuroFilter (fc=1.0Hz, beta=2.0)...")
        self.tracker = HandTracker(
            fc_min=1.0,
            beta=2.0,
            gripper_thresh=0.05,
            homing_dwell_time=0.8,
            homing_radius_px=50,
            deadman_timeout_sec=1.5,
            deadman_max_missing_frames=45,
        )

        # 3. Initialize kinematics mapping and analytical 6-DOF IK solver
        print("[3/4] Initializing KinematicsMapper & 6-DOF Analytical Inverse Kinematics...")
        self.mapper = KinematicsMapper(
            home_x=425.0,
            home_y=0.0,
            home_z=350.0,
            scale_x=850.0,
            scale_y=1100.0,
            scale_z=1350.0,
            x_min=140.0,
            x_max=720.0,
            y_min=-550.0,
            y_max=550.0,
            z_min=-20.0,
            z_max=880.0,
        )

        # 4. Initialize communication bridges to DENSO RC8 (TCP) and Isaac Sim (UDP)
        print(f"[4/4] Establishing telemetry bridge to WINCAPS III ({rc8_host}:{rc8_port}) & Isaac Sim...")
        self.bridge = WincapsBridge(
            host=rc8_host,
            port=rc8_port,
            send_rate_hz=60.0,
            auto_reconnect=True,
            reconnect_interval_sec=3.0,
        )
        self.bridge.connect()

        # Metrics and keep-alive state
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_calc_time = time.perf_counter()
        self.last_idle_send = time.perf_counter()

    def draw_hud(
        self,
        frame: np.ndarray,
        track_data: dict,
        joints: Tuple[float, float, float, float, float, float],
        gripper: int,
        cart_pose: Tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
        """Render cyber telemetry HUD overlay and anatomical hand skeleton."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # --- TELEMETRY BADGE HEADER: DEXDENSO TELEOPERATION ---
        card_w, card_h = 320, 42
        cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), self.COLOR_BG_CARD, -1)
        cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), self.COLOR_CYAN, 1)

        cv2.putText(
            frame,
            "DEXDENSO VS-6577 TELEOP",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.COLOR_CYAN,
            2,
            cv2.LINE_AA,
        )

        # Alpha blend HUD card background
        alpha = 0.65
        cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)

        # --- RENDER HAND SKELETON MESH ---
        if track_data["hand_detected"]:
            self.tracker.draw_skeleton(
                frame=frame,
                track_data=track_data,
                w=w,
                h=h,
            )

        return frame

    def run(self) -> None:
        """Main execution loop delivering deterministic 50-60 FPS teleoperation."""
        print()
        print("=" * 70)
        print("  SYSTEM READY: 6-DOF INVERSE KINEMATICS TELEOPERATION ACTIVE")
        print("  - Default Mode : Natural 3D Reach Inverse Kinematics.")
        print("  - Controls     : Present single or both hands to the camera.")
        print("  - Right Hand   : 3D Translation (Forward/Back, Left/Right, Up/Down) & Wrist orientation.")
        print("  - Left Hand    : Wide-range Base J1 rotation and Z elevation assistance.")
        print("  - Gripper      : Pinch thumb and index finger to close gripper.")
        print("  - Key [m]      : Toggle between Natural 3D Reach and Direct Joint mapping.")
        print("  - Key [r]      : Recalibrate/Homing origin. [q] or ESC: Exit application.")
        print("=" * 70)
        print()

        window_name = "DexDenso Teleoperation Bridge"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        try:
            while True:
                # 1. Read newest video frame (zero buffer accumulation)
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.001)
                    continue

                # 2. Process vision and gesture tracking
                track_data = self.tracker.process_frame(frame)

                # 3. Solve 6-DOF kinematics (Analytical Inverse Kinematics)
                j1, j2, j3, j4, j5, j6, gripper, cart_pose = self.mapper.map_hand_to_joints(track_data)
                joints = (j1, j2, j3, j4, j5, j6)

                # 4. Dispatch joint telemetry to Isaac Sim (UDP 5008/5005) and RC8 (TCP)
                if track_data["hand_detected"]:
                    self.bridge.send_joints(
                        j1=j1,
                        j2=j2,
                        j3=j3,
                        j4=j4,
                        j5=j5,
                        j6=j6,
                        gripper=gripper,
                        fk_x=cart_pose[0],
                        fk_y=cart_pose[1],
                        fk_z=cart_pose[2],
                        force=False,
                    )
                else:
                    # When hand tracking is temporarily lost (Deadman active): maintain 10Hz keep-alive
                    now_t = time.perf_counter()
                    if now_t - self.last_idle_send >= 0.10:
                        self.bridge.send_joints(
                            j1=j1,
                            j2=j2,
                            j3=j3,
                            j4=j4,
                            j5=j5,
                            j6=j6,
                            gripper=gripper,
                            fk_x=cart_pose[0],
                            fk_y=cart_pose[1],
                            fk_z=cart_pose[2],
                            force=True,
                        )
                        self.last_idle_send = now_t
                    if not self.bridge.is_connected:
                        self.bridge.start_background_connect()

                # 5. Calculate real-time FPS
                self.frame_count += 1
                now = time.perf_counter()
                if now - self.fps_calc_time >= 0.5:
                    self.fps = self.frame_count / (now - self.fps_calc_time)
                    self.frame_count = 0
                    self.fps_calc_time = now

                # 6. Render HUD telemetry overlay
                display_frame = self.draw_hud(
                    frame=track_data["mirrored_frame"],
                    track_data=track_data,
                    joints=joints,
                    gripper=gripper,
                    cart_pose=cart_pose,
                )

                # 7. Display video frame
                cv2.imshow(window_name, display_frame)

                # 8. Handle interactive keyboard commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\nUser requested shutdown...")
                    break
                elif key == ord("r"):
                    print("--> Recalibrating origin and homing benchmarks...")
                    self.tracker.reset_homing()
                    self.mapper.reset()
                elif key == ord("m"):
                    new_mode = self.mapper.toggle_mode()
                    print(f"--> [CONTROL MODE SWITCHED]: {new_mode}")
                elif key == ord("+") or key == ord("="):
                    self.mapper.scale_x = min(1500.0, self.mapper.scale_x * 1.15)
                    self.mapper.scale_y = min(1800.0, self.mapper.scale_y * 1.15)
                    self.mapper.scale_z = min(2200.0, self.mapper.scale_z * 1.15)
                    print(f"--> [SENSITIVITY INCREASED] Scales: X={self.mapper.scale_x:.0f}, Y={self.mapper.scale_y:.0f}, Z={self.mapper.scale_z:.0f}")
                elif key == ord("-") or key == ord("_"):
                    self.mapper.scale_x = max(300.0, self.mapper.scale_x * 0.85)
                    self.mapper.scale_y = max(400.0, self.mapper.scale_y * 0.85)
                    self.mapper.scale_z = max(500.0, self.mapper.scale_z * 0.85)
                    print(f"--> [SENSITIVITY DECREASED] Scales: X={self.mapper.scale_x:.0f}, Y={self.mapper.scale_y:.0f}, Z={self.mapper.scale_z:.0f}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught (Ctrl+C)...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully release hardware handles, network sockets, and GUI resources."""
        print("\n[SHUTDOWN] Releasing system resources...")
        self.bridge.close()
        self.tracker.close()
        self.camera.stop()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Camera handles, sockets, and GUI windows successfully released.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DexDenso Teleoperation Bridge")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--host", type=str, default="192.168.1.131", help="WINCAPS III RC8 IP address (default: 192.168.1.131)")
    parser.add_argument("--port", type=int, default=49152, help="RC8 TCP Server port (default: 49152)")

    args = parser.parse_args()

    app = TeleopApp(camera_id=args.camera, rc8_host=args.host, rc8_port=args.port)
    app.run()
