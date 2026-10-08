"""
Module: kinematics_mapper.py
Description: Maps hand tracking gestures from computer vision to the 6-DOF coordinate
             system and joint angles of the DENSO VS-6577 industrial robotic arm.
             Features an analytical 6-DOF Inverse Kinematics solver:
             - Natural industrial elbow flexure (anthropomorphic forward reach).
             - Upright downward tool orientation (Tool Downward constraint: Z = -1).
             - Dual-hand bimanual coordination and single-hand 3D reach modes.
             - Rate limiting and deadband filtering for zero-jitter, real-time control.
Author: Computer Vision & Robotics Teleoperation Engineering Team
"""

from typing import Tuple, Optional, Dict, Any
import math
import numpy as np


class KinematicsMapper:
    """
    6-DOF Kinematic Mapper for the DENSO VS-6577 industrial manipulator:
    - DENSO Base Frame (Right-Hand Rule):
        +X: Forward from the robot base (140.0 mm -> 720.0 mm)
        +Y: Lateral left from the robot base (-550.0 mm -> +550.0 mm)
        +Z: Vertical upward (-20.0 mm table clearance -> +880.0 mm maximum elevation)
    - 6-DOF Analytical Inverse Kinematics:
        Computes exact J1..J6 joint angles enforcing a downward-pointing end-effector (Z = -1),
        maintaining natural elbow flexion analogous to human arm reach.
    """

    def __init__(
        self,
        home_x: float = 425.0,
        home_y: float = 0.0,
        home_z: float = 350.0,
        scale_x: float = 850.0,   # mm / unit delta forward-backward (hand depth)
        scale_y: float = 1100.0,  # mm / unit delta lateral left-right
        scale_z: float = 1350.0,  # mm / unit delta vertical up-down
        x_min: float = 140.0,     # Workspace boundary for DENSO VS-6577
        x_max: float = 720.0,
        y_min: float = -550.0,
        y_max: float = 550.0,
        z_min: float = -20.0,     # Proximity to tabletop working surface
        z_max: float = 880.0,     # Full vertical reach
        rx: float = 180.0,
        ry: float = 0.0,
        rz: float = 0.0,
    ):
        self.home_x = home_x
        self.home_y = home_y
        self.home_z = home_z

        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.rx = rx
        self.ry = ry
        self.rz = rz

        # Kinematic control mode: "CARTESIAN_IK" (default) or "DIRECT_JOINT"
        self.kinematics_mode: str = "CARTESIAN_IK"

        # Safe physical joint angle limits for DENSO VS-6577 (degrees)
        self.joint_limits = [
            (-165.0, 165.0),  # J1: Base Yaw
            (-85.0, 115.0),   # J2: Shoulder Pitch
            (10.0, 160.0),    # J3: Elbow Pitch
            (-160.0, 160.0),  # J4: Forearm Roll
            (-115.0, 115.0),  # J5: Wrist Pitch
            (-180.0, 180.0),  # J6: Flange Roll
        ]

        # Precompute nominal home joint state via analytical IK
        home_ik = self.solve_ik_vs6577(self.home_x, self.home_y, self.home_z)
        self.home_j = list(home_ik)
        self.prev_joints = list(self.home_j)

        # Angular scaling factors for Direct Joint mode
        self.scale_j1 = 340.0
        self.scale_j2 = 240.0
        self.scale_j3 = 260.0
        self.scale_j4 = 300.0

        # Anti-jitter filtering and clamp telemetry
        self.clamp_hold_frames: int = 0
        self.is_clamped_state: bool = False
        self.prev_pose: Optional[Tuple[float, float, float]] = (home_x, home_y, home_z)

    def toggle_mode(self) -> str:
        """Toggle between CARTESIAN_IK and DIRECT_JOINT control modes."""
        if self.kinematics_mode == "CARTESIAN_IK":
            self.kinematics_mode = "DIRECT_JOINT"
        else:
            self.kinematics_mode = "CARTESIAN_IK"
        return self.kinematics_mode

    def reset(self) -> None:
        """Reset the kinematic mapper back to the default home configuration."""
        home_ik = self.solve_ik_vs6577(self.home_x, self.home_y, self.home_z)
        self.prev_joints = list(home_ik)
        self.prev_pose = (self.home_x, self.home_y, self.home_z)
        self.clamp_hold_frames = 0
        self.is_clamped_state = False

    @staticmethod
    def solve_ik_vs6577(
        x_mm: float,
        y_mm: float,
        z_mm: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Analytical 6-DOF Inverse Kinematics solver for DENSO VS-6577:
        - J1: Base Yaw derived from planar coordinates (X, Y).
        - J2, J3: Shoulder and Elbow pitch angles ensuring natural reach without singularity locking.
        - J4: Forearm axial rotation mapped to hand yaw.
        - J5: Flange pitch maintaining normal downward orientation (Tool Downward: Z = -1) + pitch trim.
        - J6: Flange roll mapped to hand roll.
        """
        j1_rad = math.atan2(y_mm, x_mm)
        r_xy = math.sqrt(x_mm**2 + y_mm**2)

        L6 = 80.0       # Flange offset (80 mm)
        z_wrist = z_mm + L6
        r_wrist = r_xy

        L2 = 365.0      # Upper arm length (365 mm)
        L3 = 405.0      # Forearm length (405 mm)
        d4 = 90.0       # Elbow offset (90 mm)
        L3_eff = math.sqrt(L3**2 + d4**2)
        alpha_elbow = math.atan2(d4, L3)

        dR = r_wrist - 75.0
        dZ = z_wrist - 335.0
        D = math.sqrt(dR**2 + dZ**2)
        D = max(120.0, min(D, L2 + L3_eff - 5.0))

        cos_beta = (L2**2 + L3_eff**2 - D**2) / (2.0 * L2 * L3_eff)
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))
        j3_rad = (math.pi - beta) - alpha_elbow

        gamma = math.atan2(dZ, dR)
        cos_phi = (L2**2 + D**2 - L3_eff**2) / (2.0 * L2 * D)
        phi = math.acos(max(-1.0, min(1.0, cos_phi)))
        j2_rad = math.pi / 2.0 - (gamma + phi)

        j1_deg = math.degrees(j1_rad)
        j2_deg = math.degrees(j2_rad)
        j3_deg = math.degrees(j3_rad)

        # 3-DOF spherical wrist decoupling
        j4_deg = yaw_deg

        # Tool Downward condition: j2 + j3 + j5 = 180 deg -> tool normal perpendicular to table
        nominal_j5 = 180.0 - (j2_deg + j3_deg)
        j5_deg = nominal_j5 - pitch_deg
        j5_deg = max(-115.0, min(115.0, j5_deg))

        j6_deg = roll_deg

        return (j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg)

    @staticmethod
    def forward_kinematics(
        j1_deg: float, j2_deg: float, j3_deg: float, j4_deg: float, j5_deg: float, j6_deg: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute Cartesian end-effector coordinates from 6 joint angles (Forward Kinematics)."""
        j1 = math.radians(j1_deg)
        j2 = math.radians(j2_deg)
        j3 = math.radians(j3_deg)
        j5 = math.radians(j5_deg)

        L1 = 335.0
        d1 = 75.0
        L2 = 365.0
        L3 = 405.0
        d4 = 90.0
        L6 = 80.0

        r_elbow = d1 + L2 * math.sin(j2)
        z_elbow = L1 + L2 * math.cos(j2)

        theta_arm = j2 + j3
        r_wrist = r_elbow + L3 * math.sin(theta_arm) + d4 * math.cos(theta_arm)
        z_wrist = z_elbow + L3 * math.cos(theta_arm) - d4 * math.sin(theta_arm)

        theta_tool = theta_arm + j5
        r_flange = r_wrist + L6 * math.sin(theta_tool)
        z_flange = z_wrist - L6 * math.cos(theta_tool)

        x = r_flange * math.cos(j1)
        y = r_flange * math.sin(j1)
        z = z_flange
        return (x, y, z, j6_deg, j5_deg, j4_deg)

    def map_hand_to_joints(
        self,
        track_data: Dict[str, Any],
    ) -> Tuple[float, float, float, float, float, float, int, Tuple[float, float, float, float, float, float]]:
        """
        Map tracking gestures to 6 joint targets for the DENSO VS-6577:
        - Default: Solves 3D Inverse Kinematics with tool downward constraint.
        - Supports seamless single-hand or dual-hand bimanual coordination.
        """
        is_homed = track_data.get("is_homed", False)
        mode = track_data.get("control_mode", "SINGLE_HAND")
        hand_detected = track_data.get("hand_detected", False)
        gripper = track_data.get("gripper_state", 0)

        # If not calibrated or hand lost: maintain safe freeze state
        if not is_homed or not hand_detected:
            j1, j2, j3, j4, j5, j6 = self.prev_joints
            cur_pose = self.forward_kinematics(j1, j2, j3, j4, j5, j6)
            return (j1, j2, j3, j4, j5, j6, gripper, cur_pose)

        r_roll, r_pitch, r_yaw = track_data.get("right_angles", (0.0, 0.0, 0.0))

        if self.kinematics_mode == "CARTESIAN_IK":
            # === CARTESIAN INVERSE KINEMATICS MODE ===
            if mode == "DUAL_HAND":
                # Dual-hand coordination:
                # Right hand: primary 3D Cartesian target (X, Y, Z) + wrist orientation + gripper
                # Left hand: wide-range base rotation (J1) and auxiliary Z elevation
                r_pos = track_data.get("current_right_pos") or track_data.get("right_pos")
                r_orig = track_data.get("origin_right")
                l_pos = track_data.get("current_left_pos") or track_data.get("left_pos")
                l_orig = track_data.get("origin_left")

                dx_r = (r_pos[0] - r_orig[0]) if (r_pos and r_orig) else 0.0
                dy_r = (r_pos[1] - r_orig[1]) if (r_pos and r_orig) else 0.0
                dz_r = (r_pos[2] - r_orig[2]) if (r_pos and r_orig) else 0.0

                dx_l = (l_pos[0] - l_orig[0]) if (l_pos and l_orig) else 0.0
                dy_l = (l_pos[1] - l_orig[1]) if (l_pos and l_orig) else 0.0
                dz_l = (l_pos[2] - l_orig[2]) if (l_pos and l_orig) else 0.0

                # Bimanual fusion
                raw_x = self.home_x + self.scale_x * dz_r + (0.5 * self.scale_x * dz_l)
                raw_y = self.home_y - self.scale_y * dx_r - (0.6 * self.scale_y * dx_l)
                raw_z = self.home_z - self.scale_z * dy_r - (0.6 * self.scale_z * dy_l)

            else:
                # Single-hand: full 3D Cartesian tracking
                s_pos = track_data.get("single_pos") or track_data.get("filtered_pos")
                s_orig = track_data.get("single_origin") or track_data.get("origin_pos")

                dx = (s_pos[0] - s_orig[0]) if (s_pos and s_orig) else 0.0
                dy = (s_pos[1] - s_orig[1]) if (s_pos and s_orig) else 0.0
                dz = (s_pos[2] - s_orig[2]) if (s_pos and s_orig) else 0.0

                raw_x = self.home_x + self.scale_x * dz
                raw_y = self.home_y - self.scale_y * dx
                raw_z = self.home_z - self.scale_z * dy

            # Enforce industrial safety workspace bounding box
            clamped_x = float(np.clip(raw_x, self.x_min, self.x_max))
            clamped_y = float(np.clip(raw_y, self.y_min, self.y_max))
            clamped_z = float(np.clip(raw_z, self.z_min, self.z_max))

            # Micro-deadband filter (0.8 mm) eliminating static hand tremor
            if self.prev_pose is not None:
                px, py, pz = self.prev_pose
                dist = math.hypot(math.hypot(clamped_x - px, clamped_y - py), clamped_z - pz)
                if dist < 0.8:
                    clamped_x, clamped_y, clamped_z = px, py, pz
            self.prev_pose = (clamped_x, clamped_y, clamped_z)

            # Solve analytical 6-DOF IK with tool downward orientation
            target_j1, target_j2, target_j3, target_j4, target_j5, target_j6 = self.solve_ik_vs6577(
                clamped_x, clamped_y, clamped_z, r_roll, r_pitch, r_yaw
            )
            cart_pose = (clamped_x, clamped_y, clamped_z, r_roll, r_pitch, r_yaw)

        else:
            # === DIRECT JOINT TELEOPERATION MODE ===
            left_pos = track_data.get("current_left_pos") or track_data.get("left_pos")
            origin_left = track_data.get("origin_left")
            right_pos = track_data.get("current_right_pos") or track_data.get("right_pos")
            origin_right = track_data.get("origin_right")

            dx_l = (left_pos[0] - origin_left[0]) if (left_pos and origin_left) else 0.0
            dy_l = (left_pos[1] - origin_left[1]) if (left_pos and origin_left) else 0.0
            dx_r = (right_pos[0] - origin_right[0]) if (right_pos and origin_right) else 0.0
            dy_r = (right_pos[1] - origin_right[1]) if (right_pos and origin_right) else 0.0

            # Nominal anthropomorphic pose baseline (J2=25 deg, J3=105 deg, J5=50 deg)
            target_j1 = self.home_j[0] - (self.scale_j1 * dx_l)
            target_j2 = 25.0 + (self.scale_j2 * dy_l)
            target_j3 = 105.0 + (self.scale_j3 * dy_r)
            target_j4 = - (self.scale_j4 * dx_r) + (1.2 * r_yaw)
            nominal_j5 = 180.0 - (target_j2 + target_j3)
            target_j5 = nominal_j5 - (1.2 * r_pitch)
            target_j6 = 1.2 * r_roll
            cart_pose = self.forward_kinematics(target_j1, target_j2, target_j3, target_j4, target_j5, target_j6)

        # Enforce physical mechanical joint angle limits
        target_j1 = float(np.clip(target_j1, self.joint_limits[0][0], self.joint_limits[0][1]))
        target_j2 = float(np.clip(target_j2, self.joint_limits[1][0], self.joint_limits[1][1]))
        target_j3 = float(np.clip(target_j3, self.joint_limits[2][0], self.joint_limits[2][1]))
        target_j4 = float(np.clip(target_j4, self.joint_limits[3][0], self.joint_limits[3][1]))
        target_j5 = float(np.clip(target_j5, self.joint_limits[4][0], self.joint_limits[4][1]))
        target_j6 = float(np.clip(target_j6, self.joint_limits[5][0], self.joint_limits[5][1]))

        targets = [target_j1, target_j2, target_j3, target_j4, target_j5, target_j6]
        max_step = 12.0  # Maximum smooth angular step per frame (degrees)
        new_joints = []
        for i in range(6):
            diff = targets[i] - self.prev_joints[i]
            if abs(diff) > max_step:
                diff = math.copysign(max_step, diff)
            val = self.prev_joints[i] + diff
            if abs(val - self.prev_joints[i]) < 0.15:
                val = self.prev_joints[i]
            new_joints.append(val)

        self.prev_joints = new_joints
        j1, j2, j3, j4, j5, j6 = new_joints

        return (j1, j2, j3, j4, j5, j6, gripper, cart_pose)

    def map_hand_to_robot(
        self,
        current_pos: Optional[Tuple[float, float, float]],
        origin_pos: Optional[Tuple[float, float, float]],
        is_homed: bool,
    ) -> Tuple[float, float, float, float, float, float, bool]:
        """Backward compatibility interface for legacy Cartesian delta callers."""
        if not is_homed or origin_pos is None:
            return (self.home_x, self.home_y, self.home_z, self.rx, self.ry, self.rz, False)

        if current_pos is None:
            hx, hy, hz = self.prev_pose if self.prev_pose else (self.home_x, self.home_y, self.home_z)
            return (hx, hy, hz, self.rx, self.ry, self.rz, False)

        cur_x, cur_y, cur_scale = current_pos
        orig_x, orig_y, orig_scale = origin_pos

        raw_x = self.home_x + self.scale_x * (cur_scale - orig_scale)
        raw_y = self.home_y - self.scale_y * (cur_x - orig_x)
        raw_z = self.home_z - self.scale_z * (cur_y - orig_y)

        robot_x = float(np.clip(raw_x, self.x_min, self.x_max))
        robot_y = float(np.clip(raw_y, self.y_min, self.y_max))
        robot_z = float(np.clip(raw_z, self.z_min, self.z_max))

        clamped = (raw_x != robot_x) or (raw_y != robot_y) or (raw_z != robot_z)
        self.prev_pose = (robot_x, robot_y, robot_z)
        return (robot_x, robot_y, robot_z, self.rx, self.ry, self.rz, clamped)

    def get_workspace_bounds(self) -> Dict[str, Any]:
        """Retrieve workspace safety boundary parameters for telemetry HUD rendering."""
        return {
            "x_bounds": (self.x_min, self.x_max),
            "y_bounds": (self.y_min, self.y_max),
            "z_bounds": (self.z_min, self.z_max),
            "home": (self.home_x, self.home_y, self.home_z),
        }
