"""
Standalone IK test — validates inverse kinematics without needing a camera.

Steps through hardcoded target positions, solves IK for each, moves the arm,
and prints the actual gripper position vs. the target for verification.

Usage:
    python test_ik.py
"""

import os
import time

import mujoco
import mujoco.viewer

from ik_solver import solve_ik
from utils import move_to_pose, hold_position, get_gripper_position


# Hardcoded test targets: (x, y, z) in meters
TEST_TARGETS = [
    [0.15, 0.0, 0.15],    # Front center, raised
    [0.10, 0.10, 0.10],   # Front-left, lower
    [0.10, -0.10, 0.10],  # Front-right, lower
    [0.20, 0.0, 0.05],    # Far forward, near table
    [0.0, 0.15, 0.20],    # Left side, raised
]


def main():
    model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    print("=== SO-101 IK Test ===")
    print(f"Testing {len(TEST_TARGETS)} target positions\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Let the simulation settle
        for _ in range(100):
            mujoco.mj_step(m, d)
        viewer.sync()

        for i, target in enumerate(TEST_TARGETS):
            print(f"--- Target {i+1}/{len(TEST_TARGETS)}: {target} ---")

            # Solve IK
            joint_angles = solve_ik(m, d, target, max_iterations=300, tolerance=0.001)
            print(f"  IK angles: { {k: f'{v:.1f}°' for k, v in joint_angles.items()} }")

            # Move arm to solved position
            move_to_pose(m, d, viewer, joint_angles, duration_sec=2.0)

            # Hold and check
            hold_position(m, d, viewer, duration_sec=1.0)

            # Read actual gripper position
            actual = get_gripper_position(m, d)
            error = sum((a - t) ** 2 for a, t in zip(actual, target)) ** 0.5
            print(f"  Actual:  [{actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f}]")
            print(f"  Target:  [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
            print(f"  Error:   {error*1000:.1f} mm")

            if error < 0.005:
                print("  Result:  PASS")
            else:
                print("  Result:  FAIL (>5mm)")
            print()

            if not viewer.is_running():
                break

        # Keep viewer open after tests
        print("Tests complete. Close the viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
