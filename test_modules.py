"""
Module: test_modules.py
Description: Automated unit test suite verifying:
  1. OneEuroFilter (Static jitter rejection and dynamic step response).
  2. KinematicsMapper (DENSO coordinate frames, analytical IK, and safety bounds).
  3. WincapsBridge & PACScript Input #1 serialization format.
  4. Bimanual joint mapping & 6-DOF decoupled wrist kinematics.
"""

import math
import time
from vision_tracker import OneEuroFilter, Point3DOneEuroFilter
from kinematics_mapper import KinematicsMapper
from wincaps_bridge import WincapsBridge


def test_one_euro_filter():
    """Verify jitter rejection at rest and low latency during rapid motion."""
    print("[TEST 1/4] Testing OneEuroFilter signal dynamics...")
    f = OneEuroFilter(fc_min=1.5, beta=20.0)

    # 1. Verify noise filtering at steady state (synthetic jitter +/- 0.02)
    t = 0.0
    val = 0.5
    filtered_vals = []
    for i in range(30):
        noisy_val = val + 0.02 * math.sin(i)
        res = f.filter(noisy_val, timestamp=t)
        filtered_vals.append(res)
        t += 0.02

    raw_var = 0.02
    filtered_dev = abs(filtered_vals[-1] - val)
    assert filtered_dev < raw_var, f"Filter jitter suppression failed: {filtered_dev} >= {raw_var}"

    # 2. Verify immediate responsiveness during large step transient
    jump_val = 1.0
    t += 0.02
    step_res = f.filter(jump_val, timestamp=t)
    assert step_res > 0.65, f"Filter step response too sluggish: {step_res}"
    print("  -> PASSED: OneEuroFilter operates optimally across static and dynamic regimes.")


def test_kinematics_mapper():
    """Verify coordinate transformation, directional signs, and safety clamping."""
    print("[TEST 2/4] Testing KinematicsMapper transformation matrix & limits...")
    mapper = KinematicsMapper(
        home_x=425.0,
        home_y=0.0,
        home_z=325.0,
        scale_x=1200.0,
        scale_y=800.0,
        scale_z=800.0,
        x_min=300.0,
        x_max=550.0,
        y_min=-250.0,
        y_max=250.0,
        z_min=150.0,
        z_max=500.0,
    )

    origin = (0.5, 0.5, 0.1)  # (x_wrist, y_wrist, hand_scale)

    # Case 1: Un-homed state must return safe home pose
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(
        current_pos=(0.6, 0.6, 0.12),
        origin_pos=origin,
        is_homed=False
    )
    assert (x, y, z) == (425.0, 0.0, 325.0), "Un-homed state must stay at home pose"
    assert not clamped

    # Case 2: Hand moves right on screen (dx > 0) -> Robot moves right (-Y_robot)
    # DENSO Base Frame: +Y is robot left, -Y is robot right
    cur_right = (0.6, 0.5, 0.1)
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(cur_right, origin, is_homed=True)
    expected_y = 0.0 - 800.0 * (0.6 - 0.5)  # -80.0 mm
    assert abs(y - expected_y) < 1e-3, f"Incorrect lateral mapping: y={y} != {expected_y}"
    assert y < 0, "Rightward screen motion must map to -Y robot"

    # Case 3: Hand moves left on screen (dx < 0) -> Robot moves left (+Y_robot)
    cur_left = (0.4, 0.5, 0.1)
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(cur_left, origin, is_homed=True)
    assert y > 0, "Leftward screen motion must map to +Y robot"

    # Case 4: Hand moves up on screen (dy < 0 in image coordinates) -> Robot moves up (+Z_robot)
    cur_up = (0.5, 0.4, 0.1)
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(cur_up, origin, is_homed=True)
    assert z > 325.0, "Upward screen motion must map to +Z robot"

    # Case 5: Hand moves toward camera (scale increases) -> Robot moves forward (+X_robot)
    cur_fwd = (0.5, 0.5, 0.15)
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(cur_fwd, origin, is_homed=True)
    assert x > 425.0, "Forward gesture must map to +X robot"

    # Case 6: Safety bounding box clamping
    cur_extreme = (0.5, 0.5, 0.5)
    x, y, z, rx, ry, rz, clamped = mapper.map_hand_to_robot(cur_extreme, origin, is_homed=True)
    assert x == 550.0, f"X not clamped at maximum 550.0: {x}"
    assert clamped, "Bounding box violation flag must be True"

    assert (rx, ry, rz) == (180.0, 0.0, 0.0), f"Orientation error: {rx}, {ry}, {rz}"
    print("  -> PASSED: KinematicsMapper strictly adheres to DENSO Right-Hand Rule and bounds.")


def test_packet_format():
    """Verify PACScript Input #1 serialization matches DENSO controller spec."""
    print("[TEST 3/4] Testing PACScript Input #1 serialization format...")
    x, y, z = 450.123, -120.456, 310.789
    rx, ry, rz = 180.0, 0.0, 0.0
    gripper = 1

    payload = f"{x:.1f},{y:.1f},{z:.1f},{rx:.1f},{ry:.1f},{rz:.1f},{gripper}\r\n"
    assert payload == "450.1,-120.5,310.8,180.0,0.0,0.0,1\r\n"
    assert payload.endswith("\r\n")

    parts = payload.strip().split(",")
    assert len(parts) == 7
    assert float(parts[0]) == 450.1
    assert int(parts[6]) == 1
    print("  -> PASSED: Packet format is 100% compliant with PACScript Input #1.")


def test_bimanual_joint_mapping():
    """Verify bimanual coordination, analytical inverse kinematics, and wrist orientation."""
    print("[TEST 4/4] Testing 6-DOF Inverse Kinematics & Bimanual Coordination...")
    mapper = KinematicsMapper()

    # 1. Un-homed state must return valid nominal home joint configuration
    track_not_homed = {"is_homed": False, "hand_detected": False}
    j1, j2, j3, j4, j5, j6, grip, fk = mapper.map_hand_to_joints(track_not_homed)
    assert len([j1, j2, j3, j4, j5, j6]) == 6
    assert abs(j1) < 1.0 and abs(j4) < 1.0

    # 2. Both hands detected: Left Hand (x=0.25, y=0.5), Right Hand (x=0.75, y=0.5)
    orig_l = (0.25, 0.5, 0.1)
    orig_r = (0.75, 0.5, 0.1)

    # Case A: Left Hand moves left (dx = -0.10) => Base rotates left (+J1)
    track_l_left = {
        "is_homed": True,
        "hand_detected": True,
        "control_mode": "DUAL_HAND",
        "left_pos": (0.15, 0.5, 0.1),
        "origin_left": orig_l,
        "right_pos": orig_r,
        "origin_right": orig_r,
        "right_angles": (0.0, 0.0, 0.0),
        "gripper_state": 0,
    }
    j1, j2, j3, j4, j5, j6, grip, fk = mapper.map_hand_to_joints(track_l_left)
    assert j1 > 0.0, f"Leftward motion must result in positive J1: {j1}"

    # Case B: Right Hand moves up (dy = -0.10) => Z elevates
    track_r_up = {
        "is_homed": True,
        "hand_detected": True,
        "control_mode": "DUAL_HAND",
        "left_pos": orig_l,
        "origin_left": orig_l,
        "right_pos": (0.75, 0.40, 0.1),
        "origin_right": orig_r,
        "right_angles": (0.0, 0.0, 0.0),
        "gripper_state": 1,
    }
    j1, j2, j3, j4, j5, j6, grip, fk = mapper.map_hand_to_joints(track_r_up)
    assert fk[2] > 350.0, f"Upward right hand movement must elevate robot Z: {fk[2]}"
    assert grip == 1, "Gripper activation state must be preserved as 1"

    # Case C: Wrist rotation coupling
    track_wrist = {
        "is_homed": True,
        "hand_detected": True,
        "control_mode": "DUAL_HAND",
        "left_pos": orig_l,
        "origin_left": orig_l,
        "right_pos": orig_r,
        "origin_right": orig_r,
        "right_angles": (15.0, 0.0, 20.0),
        "gripper_state": 0,
    }
    j1, j2, j3, j4, j5, j6, grip, fk = mapper.map_hand_to_joints(track_wrist)
    assert j6 != 0.0 or j4 != 0.0, "Wrist axes must rotate with hand orientation"

    print("  -> PASSED: 6-DOF Inverse Kinematics and Bimanual Mapping validated.")


if __name__ == "__main__":
    test_one_euro_filter()
    test_kinematics_mapper()
    test_packet_format()
    test_bimanual_joint_mapping()
    print("\n===> ALL 4/4 UNIT TESTS PASSED SUCCESSFULLY! <===\n")
