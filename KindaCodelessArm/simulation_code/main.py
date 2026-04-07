"""
SO-101 Vision Pipeline — Single entry point.

Simulation mode: renders from MuJoCo overhead camera, commands MuJoCo actuators.
Real mode:       reads from USB webcam, sends commands via LeRobot.

Usage:
    python main.py                  # uses config.yaml defaults
    python main.py --mode real      # override mode from command line
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import yaml

from lerobot.motors import Motor, MotorNormMode

from vision import (
    capture_frame_sim,
    capture_frame_real,
    detect_object,
    pixel_to_world,
    draw_detection,
)
from ik_solver import solve_ik
from utils import send_joint_command, move_to_pose, get_gripper_position, get_current_joint_angles

#Default configs pulled from keyboard_demo.py
DEFAULT_MOTORS = {
                "joint_1": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "joint_2": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "joint_3": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "joint_4": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "joint_5": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "joint_6": Motor(6, "sts3215", MotorNormMode.DEGREES),
            }

DEFAULT_PORT = '/dev/ttyACM0'


def load_config(path: str = "config.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(__file__), path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def show_preview(window_name: str, frame, preview_enabled: bool) -> tuple[bool, bool]:
    """
    Show an OpenCV preview window when enabled.

    Returns:
        (preview_enabled, should_quit)
    """
    if not preview_enabled:
        return False, False

    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return True, key == ord("q")
    except cv2.error as exc:
        print(f"[WARN] Disabling OpenCV preview window: {exc}")
        print("       Simulation will continue in the MuJoCo viewer only.")
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
        return False, False


def load_homography(config: dict) -> np.ndarray:
    hpath = os.path.join(os.path.dirname(__file__), config["homography"]["matrix_file"])
    if not os.path.exists(hpath):
        print(f"[WARN] Homography file not found at {hpath}")
        print("       Run 'python calibration/calibrate.py' first.")
        print("       Using identity (pixel = world) as fallback — results will be wrong.")
        return np.eye(3)
    return np.load(hpath)


def move_gripper_to_coordinate(
    m,
    d,
    viewer,
    target_xyz: list[float],
    gripper_open_deg: float,
    approach_duration_sec: float = 2.0,
    gripper_duration_sec: float = 0.75,
    max_iterations: int = 200,
    tolerance: float = 0.001,
) -> dict:
    """
    Move the end-effector to target_xyz, then open the gripper.

    The arm motion and gripper motion are split into two stages so the gripper
    does not open until the target position has been reached.
    """
    current_joint_angles = get_current_joint_angles(m, d)

    arm_target = solve_ik(
        m,
        d,
        target_xyz,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_angles_deg=current_joint_angles,
    )

    arm_target["gripper"] = current_joint_angles["gripper"]
    move_to_pose(m, d, viewer, arm_target, duration_sec=approach_duration_sec)

    final_target = dict(arm_target)
    final_target["gripper"] = max(-10.0, min(100.0, gripper_open_deg))
    move_to_pose(m, d, viewer, final_target, duration_sec=gripper_duration_sec)

    return final_target


def pick_and_place_at_coordinates(
    m,
    d,
    viewer,
    pick_xy: list[float],
    place_xy: list[float],
    pick_approach_z: float,
    pick_z: float,
    carry_z: float,
    place_approach_z: float,
    place_z: float,
    gripper_open_deg: float,
    gripper_closed_deg: float,
    move_duration_sec: float = 1.5,
    grip_duration_sec: float = 0.75,
    max_iterations: int = 200,
    tolerance: float = 0.001,
    wrist_roll_deg: float | None = None,
) -> dict:
    """
    Execute a simple sideways pick-and-place sequence.
    """
    current_joint_angles = get_current_joint_angles(m, d)

    def solve_stage(target_xyz: list[float], seed_angles: dict[str, float]) -> dict[str, float]:
        return solve_ik(
            m,
            d,
            target_xyz,
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_angles_deg=seed_angles,
        )

    open_deg = max(-10.0, min(100.0, gripper_open_deg))
    closed_deg = max(-10.0, min(100.0, gripper_closed_deg))

    stage1 = solve_stage([pick_xy[0], pick_xy[1], pick_approach_z], current_joint_angles)
    if wrist_roll_deg is not None:
        stage1["wrist_roll"] = wrist_roll_deg
    stage1["gripper"] = open_deg
    move_to_pose(m, d, viewer, stage1, duration_sec=move_duration_sec)

    stage2 = solve_stage([pick_xy[0], pick_xy[1], pick_z], stage1)
    if wrist_roll_deg is not None:
        stage2["wrist_roll"] = wrist_roll_deg
    stage2["gripper"] = open_deg
    move_to_pose(m, d, viewer, stage2, duration_sec=move_duration_sec)

    stage3 = dict(stage2)
    if wrist_roll_deg is not None:
        stage3["wrist_roll"] = wrist_roll_deg
    stage3["gripper"] = closed_deg
    move_to_pose(m, d, viewer, stage3, duration_sec=grip_duration_sec)

    stage4 = solve_stage([pick_xy[0], pick_xy[1], carry_z], stage3)
    if wrist_roll_deg is not None:
        stage4["wrist_roll"] = wrist_roll_deg
    stage4["gripper"] = closed_deg
    move_to_pose(m, d, viewer, stage4, duration_sec=move_duration_sec)

    stage5 = solve_stage([place_xy[0], place_xy[1], place_approach_z], stage4)
    if wrist_roll_deg is not None:
        stage5["wrist_roll"] = wrist_roll_deg
    stage5["gripper"] = closed_deg
    move_to_pose(m, d, viewer, stage5, duration_sec=move_duration_sec)

    stage6 = solve_stage([place_xy[0], place_xy[1], place_z], stage5)
    if wrist_roll_deg is not None:
        stage6["wrist_roll"] = wrist_roll_deg
    stage6["gripper"] = closed_deg
    move_to_pose(m, d, viewer, stage6, duration_sec=move_duration_sec)

    stage7 = dict(stage6)
    if wrist_roll_deg is not None:
        stage7["wrist_roll"] = wrist_roll_deg
    stage7["gripper"] = open_deg
    move_to_pose(m, d, viewer, stage7, duration_sec=grip_duration_sec)

    stage8 = solve_stage([place_xy[0], place_xy[1], place_approach_z], stage7)
    if wrist_roll_deg is not None:
        stage8["wrist_roll"] = wrist_roll_deg
    stage8["gripper"] = open_deg
    move_to_pose(m, d, viewer, stage8, duration_sec=move_duration_sec)

    return stage8


def run_simulation(config: dict):
    """Main loop for simulation mode."""
    model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    cam_cfg = config["camera"]
    ik_cfg = config["ik"]
    manual_target_cfg = config["manual_grasp_target"]
    homography = load_homography(config)
    preview_enabled = config.get("preview_window", True)
    manual_grasp_requested = {"pending": False}

    def on_key(keycode: int) -> None:
        if keycode in (ord("g"), ord("G")):
            manual_grasp_requested["pending"] = True

    print("[SIM] Starting simulation pipeline. Press 'g' in the MuJoCo viewer to run the manual grasp sequence.")
    print("[SIM] Press 'q' in the camera window to quit.")

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        while viewer.is_running():
            # Step physics
            mujoco.mj_step(m, d)
            viewer.sync()

            if manual_grasp_requested["pending"]:
                manual_grasp_requested["pending"] = False
                pick_xy = [
                    manual_target_cfg["x"],
                    manual_target_cfg["y"],
                ]
                place_xy = [
                    manual_target_cfg["place_x"],
                    manual_target_cfg["place_y"],
                ]
                print(
                    "[SIM] Manual pick-and-place requested: "
                    f"pick=({pick_xy[0]:.3f}, {pick_xy[1]:.3f}, {manual_target_cfg['pick_z']:.3f}), "
                    f"place=({place_xy[0]:.3f}, {place_xy[1]:.3f}, {manual_target_cfg['place_z']:.3f}), "
                    f"wrist_roll={manual_target_cfg.get('wrist_roll_deg', 90.0):.1f}deg"
                )
                pick_and_place_at_coordinates(
                    m,
                    d,
                    viewer,
                    pick_xy=pick_xy,
                    place_xy=place_xy,
                    pick_approach_z=manual_target_cfg["pick_approach_z"],
                    pick_z=manual_target_cfg["pick_z"],
                    carry_z=manual_target_cfg["carry_z"],
                    place_approach_z=manual_target_cfg["place_approach_z"],
                    place_z=manual_target_cfg["place_z"],
                    gripper_open_deg=manual_target_cfg["gripper_open_deg"],
                    gripper_closed_deg=manual_target_cfg["gripper_closed_deg"],
                    move_duration_sec=1.5,
                    grip_duration_sec=0.75,
                    max_iterations=ik_cfg["max_iterations"],
                    tolerance=ik_cfg["tolerance"],
                    wrist_roll_deg=manual_target_cfg.get("wrist_roll_deg", 90.0),
                )

            # Capture overhead frame
            frame = capture_frame_sim(
                m, d,
                cam_cfg["mujoco_camera"],
                cam_cfg["width"],
                cam_cfg["height"],
            )

            # Detect object
            detection = detect_object(frame, config)

            if detection is not None:
                # Convert pixel to world coordinates
                world_xy = pixel_to_world(detection, homography)
                annotated = draw_detection(frame, detection, world_xy)

                # Build 3D target: (world_x, world_y, approach_height)
                target_xyz = [world_xy[0], world_xy[1], ik_cfg["approach_height"]]
                print(f"[SIM] Detected at px={detection}, world=({world_xy[0]:.3f}, {world_xy[1]:.3f})")

                # Solve IK
                joint_angles = solve_ik(
                    m, d, target_xyz,
                    max_iterations=ik_cfg["max_iterations"],
                    tolerance=ik_cfg["tolerance"],
                )
                print(f"[SIM] IK solution: { {k: f'{v:.1f}°' for k, v in joint_angles.items()} }")

                # Move arm
                move_to_pose(m, d, viewer, joint_angles, duration_sec=2.0)
            else:
                annotated = frame

            # Show camera feed when available; macOS OpenCV GUI can fail under mjpython.
            preview_enabled, should_quit = show_preview(
                "SO-101 Overhead",
                annotated,
                preview_enabled,
            )
            if should_quit:
                break

    cv2.destroyAllWindows()


def run_real(config: dict):
    """Main loop for real hardware mode."""
    cam_cfg = config["camera"]
    ik_cfg = config["ik"]
    hw_cfg = config["real_hardware"]
    homography = load_homography(config)
    preview_enabled = config.get("preview_window", True)

    # Open USB webcam
    cap = cv2.VideoCapture(cam_cfg["usb_device"])
    if not cap.isOpened():
        print(f"[REAL] Cannot open webcam at device index {cam_cfg['usb_device']}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    # Load MuJoCo model just for IK computation (not for rendering)
    model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Lazy import — LeRobot only needed in real mode
    try:
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
    except ImportError:
        try:
            from lerobot.motors.feetech import FeetechMotorsBus #4/6/26 Added working import for FeetechMotorsBus from keyboard_demo.py
        except ImportError:
            print("[REAL] LeRobot not installed. Install with: pip install lerobot")
            sys.exit(1)

    motors = FeetechMotorsBus(port=DEFAULT_PORT, motors=DEFAULT_MOTORS) #4/6/26 - Changed from config inputs to default inputs based on what had worked in the past
    motors.connect()

    print("[REAL] Starting real hardware pipeline. Press 'q' to quit.")

    while True:
        frame = capture_frame_real(cap)
        if frame is None:
            continue

        detection = detect_object(frame, config)

        if detection is not None:
            world_xy = pixel_to_world(detection, homography)
            annotated = draw_detection(frame, detection, world_xy)

            target_xyz = [world_xy[0], world_xy[1], ik_cfg["approach_height"]]
            print(f"[REAL] Detected at px={detection}, world=({world_xy[0]:.3f}, {world_xy[1]:.3f})")

            joint_angles = solve_ik(
                m, d, target_xyz,
                max_iterations=ik_cfg["max_iterations"],
                tolerance=ik_cfg["tolerance"],
            )
            print(f"[REAL] IK solution: { {k: f'{v:.1f}°' for k, v in joint_angles.items()} }")

            # Send to real motors via LeRobot
            # Map joint names to LeRobot motor IDs (adjust mapping as needed)
            motor_positions = { #4/6/26 - Running this code gave a TypeError since FeetechMotorsBus does some bitwise or-ing, which must involve ints.  Typecasted to suppress error.
                "joint_1": int(joint_angles["shoulder_pan"]),
                "joint_2": int(joint_angles["shoulder_lift"]),
                "joint_3": int(joint_angles["elbow_flex"]),
                "joint_4": int(joint_angles["wrist_flex"]),
                "joint_5": int(joint_angles["wrist_roll"]),
                "joint_6": int(joint_angles["gripper"]),
            }
            print(f'Motor_positions dict: {motor_positions}')
            motors.sync_write("Goal_Position", motor_positions, normalize=False)
        else:
            annotated = frame

        preview_enabled, should_quit = show_preview(
            "SO-101 Overhead",
            annotated,
            preview_enabled,
        )
        if should_quit:
            break

    cap.release()
    motors.disconnect()
    cv2.destroyAllWindows()


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="SO-101 Vision Pipeline")
    parser.add_argument(
        "--mode",
        choices=["simulation", "real"],
        default=config["mode"],
        help="Run in simulation or real hardware mode",
    )
    args = parser.parse_args()

    # Override config with CLI arg
    config["mode"] = args.mode
    print(f"[INIT] Mode: {config['mode']}")

    if config["mode"] == "simulation":
        run_simulation(config)
    else:
        run_real(config)
        """
            # ~ As of 4/6/26, running this code in real mode just made the arm make a large movement that ended up ripping the shoulder_lift motor
            out of the socket.  Additionally, we noted that this code needs cleanup code like in keyboard_demo.py that will ensure the motors aren't
            holding their position should the program fail/keyboard interrupt.  Not sure if the movement discrepancy is a result of configuration issues
            or IK issues.
        """


if __name__ == "__main__":
    main()
