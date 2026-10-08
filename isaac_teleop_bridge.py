r"""
Module: isaac_teleop_bridge.py
Description: High-Performance 6-DOF Teleoperation UDP Bridge for DENSO VS-6577 in NVIDIA Isaac Sim.
Features:
  - Dual UDP listener on primary port 5008 (with port 5005 fallback).
  - Lightweight physics target drive integration using Omniverse USD APIs.
  - Zero rendering disruption, zero backbuffer / swapchain stalls.
"""

import sys
import math
import socket
import builtins
import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# 1. Clean up existing receiver instance and socket if present
if hasattr(builtins, "__isaac_receiver__") and builtins.__isaac_receiver__:
    try:
        builtins.__isaac_receiver__.stop()
    except Exception:
        pass
    builtins.__isaac_receiver__ = None

UDP_IP = "127.0.0.1"
PRIMARY_PORT = 5008
FALLBACK_PORT = 5005


def solve_ik_vs6577(x_mm, y_mm, z_mm, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """Analytical Inverse Kinematics for DENSO VS-6577 (6-DOF) with downward tool normal."""
    j1 = math.atan2(y_mm, x_mm)
    r_xy = math.sqrt(x_mm**2 + y_mm**2)

    L6 = 80.0       # Flange offset (mm)
    z_wrist = z_mm + L6
    r_wrist = r_xy

    L2 = 365.0      # Upper arm (mm)
    L3 = 405.0      # Forearm (mm)
    d4 = 90.0       # Elbow offset (mm)
    L3_eff = math.sqrt(L3**2 + d4**2)
    alpha_elbow = math.atan2(d4, L3)

    dR = r_wrist - 75.0
    dZ = z_wrist - 335.0
    D = math.sqrt(dR**2 + dZ**2)
    D = max(120.0, min(D, L2 + L3_eff - 5.0))

    cos_beta = (L2**2 + L3_eff**2 - D**2) / (2.0 * L2 * L3_eff)
    beta = math.acos(max(-1.0, min(1.0, cos_beta)))
    j3 = (math.pi - beta) - alpha_elbow

    gamma = math.atan2(dZ, dR)
    cos_phi = (L2**2 + D**2 - L3_eff**2) / (2.0 * L2 * D)
    phi = math.acos(max(-1.0, min(1.0, cos_phi)))
    j2 = math.pi / 2.0 - (gamma + phi)

    j4 = math.radians(yaw_deg)
    raw_j5 = (math.pi - (j2 + j3)) - math.radians(pitch_deg)
    j5 = max(-math.radians(120.0), min(math.radians(120.0), raw_j5))
    j6 = math.radians(roll_deg)

    return [j1, j2, j3, j4, j5, j6]


class IsaacDensoTeleopReceiver:
    """UDP Teleoperation receiver integrated into NVIDIA Isaac Sim update loop."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass

        self.active_port = PRIMARY_PORT
        try:
            self.sock.bind((UDP_IP, PRIMARY_PORT))
        except Exception as e:
            carb.log_warn(f"[Isaac Teleop] Port {PRIMARY_PORT} busy: {e}, falling back to {FALLBACK_PORT}")
            try:
                self.sock.bind((UDP_IP, FALLBACK_PORT))
                self.active_port = FALLBACK_PORT
            except Exception as e2:
                carb.log_error(f"[Isaac Teleop] Cannot bind socket: {e2}")

        self.sock.setblocking(False)
        self.sub = None
        self.joint_prims = {}
        self.packet_count = 0

        self._find_robot_joints()
        self._start_listening()

        print("=" * 60)
        print("=== [ISAAC SIM 6-DOF TELEOP RECEIVER READY] ===")
        print(f"Listening on UDP {UDP_IP}:{self.active_port}")
        print(f"Robot joints detected ({len(self.joint_prims)}/6): {list(self.joint_prims.keys())}")
        print("-> Press Play in Isaac Sim to enable physics drives!")
        print("=" * 60)

    def _find_robot_joints(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        self.joint_prims.clear()
        target_names = [f"joint_{i}" for i in range(1, 7)]

        for prim in stage.Traverse():
            p_name = prim.GetName()
            if p_name in target_names:
                self.joint_prims[p_name] = prim

    def _start_listening(self):
        app = omni.kit.app.get_app()
        try:
            self.sub = app.get_update_event_stream().create_subscription_to_pop(self._on_update)
        except Exception:
            self.sub = app.get_update_event_stream().create_subscription_to_push(self._on_update)

    def _on_update(self, event):
        if not self.sock:
            return

        if len(self.joint_prims) < 6:
            self._find_robot_joints()

        latest_data = None
        while True:
            try:
                data, _ = self.sock.recvfrom(512)
                latest_data = data
            except (BlockingIOError, socket.error, OSError):
                break
            except Exception:
                break

        if latest_data:
            try:
                text = latest_data.decode("ascii").strip()
                tokens = [t.strip() for t in text.split(",") if t.strip()]
                if not tokens:
                    return

                if tokens[0].upper() == "JOINT":
                    if len(tokens) >= 7:
                        j_angles = [float(tokens[i]) for i in range(1, 7)]
                        grip = int(tokens[7]) if len(tokens) >= 8 else 0
                        self._apply_joint_angles(j_angles, grip)
                        self.packet_count += 1
                        if self.packet_count == 1 or self.packet_count % 60 == 0:
                            print(f"[Isaac Teleop] JOINT packet #{self.packet_count}: {[round(a, 1) for a in j_angles]}")
                elif tokens[0].upper() == "CART":
                    if len(tokens) >= 7:
                        x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                        rx, ry, rz = float(tokens[4]), float(tokens[5]), float(tokens[6])
                        grip = int(tokens[7]) if len(tokens) >= 8 else 0
                        self._apply_teleop_pose(x, y, z, rx, ry, rz, grip)
                        self.packet_count += 1
                        if self.packet_count == 1 or self.packet_count % 60 == 0:
                            print(f"[Isaac Teleop] CART packet #{self.packet_count}: ({x:.0f}, {y:.0f}, {z:.0f})")
                else:
                    vals = [float(t) for t in tokens]
                    if len(vals) >= 6:
                        grip = int(vals[6]) if len(vals) >= 7 else 0
                        if vals[2] > 180.0:
                            self._apply_teleop_pose(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], grip)
                        else:
                            self._apply_joint_angles(vals[:6], grip)
                        self.packet_count += 1
            except Exception:
                pass

    def _apply_joint_angles(self, angles_deg, grip):
        for idx in range(1, 7):
            deg_val = angles_deg[idx - 1]
            j_name = f"joint_{idx}"
            prim = self.joint_prims.get(j_name)

            if prim and prim.IsValid():
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive_api:
                    target_attr = drive_api.GetTargetPositionAttr()
                    if target_attr and target_attr.IsValid():
                        target_attr.Set(deg_val)
                    else:
                        prim.GetAttribute("drive:angular:physics:targetPosition").Set(deg_val)
                else:
                    attr = prim.GetAttribute("drive:angular:physics:targetPosition")
                    if attr and attr.IsValid():
                        attr.Set(deg_val)

                state_attr = prim.GetAttribute("state:angular:physics:position")
                if state_attr and state_attr.IsValid():
                    state_attr.Set(deg_val)

    def _apply_teleop_pose(self, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, grip):
        angles_rad = solve_ik_vs6577(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)
        angles_deg = [math.degrees(a) for a in angles_rad]
        self._apply_joint_angles(angles_deg, grip)

    def stop(self):
        if self.sub:
            self.sub = None
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        print("[Isaac Teleop] Teleoperation receiver stopped.")


# Initialize and store receiver instance in builtins
builtins.__isaac_receiver__ = IsaacDensoTeleopReceiver()
