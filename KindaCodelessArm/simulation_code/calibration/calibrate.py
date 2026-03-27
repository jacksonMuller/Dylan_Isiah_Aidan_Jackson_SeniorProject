"""
Homography Calibration Tool for SO-101 overhead camera.

Lets the user click known points in the camera frame and enter their
real-world (x, y) coordinates to compute a pixel-to-world homography matrix.

Simulation mode:  renders a frame from MuJoCo's overhead camera.
Real mode:        captures a frame from the USB webcam.

Usage:
    python calibration/calibrate.py                  # simulation mode
    python calibration/calibrate.py --mode real      # real webcam

Flow:
    Phase 1 — Click at least 4 points in the camera window, then press ENTER.
    Phase 2 — Type the real-world (x, y) for each point in the terminal.
    The homography is computed and saved automatically.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision import capture_frame_sim, capture_frame_real


# Globals for mouse callback
clicked_pixels = []
frame_display = None


def mouse_callback(event, x, y, flags, param):
    global frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixels.append((x, y))
        n = len(clicked_pixels)
        print(f"  Clicked point #{n}: ({x}, {y})  [total: {n}]")
        # Draw the click on the display
        cv2.drawMarker(frame_display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(frame_display, str(n), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Calibration", frame_display)


def get_frame(config: dict, mode: str) -> np.ndarray:
    """Capture a single frame based on mode."""
    cam_cfg = config["camera"]

    if mode == "simulation":
        import mujoco
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
        m = mujoco.MjModel.from_xml_path(model_path)
        d = mujoco.MjData(m)
        for _ in range(50):
            mujoco.mj_forward(m, d)
        frame = capture_frame_sim(m, d, cam_cfg["mujoco_camera"], cam_cfg["width"], cam_cfg["height"])
    else:
        cap = cv2.VideoCapture(cam_cfg["usb_device"])
        if not cap.isOpened():
            print(f"Cannot open webcam at device {cam_cfg['usb_device']}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame from webcam")
            sys.exit(1)

    return frame


def main():
    global frame_display

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="SO-101 Homography Calibration")
    parser.add_argument("--mode", choices=["simulation", "real"],
                        default=config["mode"])
    args = parser.parse_args()

    print("=== SO-101 Homography Calibration ===")
    print(f"Mode: {args.mode}\n")

    frame = get_frame(config, args.mode)
    frame_display = frame.copy()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", frame_display)

    # ── Phase 1: collect clicks in the OpenCV window ──
    print("PHASE 1: Click on at least 4 known points in the camera window.")
    print("         Press ENTER in this terminal when done clicking.")
    print("         Press 'q' in the camera window to quit without saving.\n")

    # Pump the OpenCV event loop until the user presses ENTER in the terminal
    # We use a short waitKey so clicks are registered, and check stdin for ENTER.
    import select
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            print("\nQuitting without saving.")
            cv2.destroyAllWindows()
            return

        # Check if ENTER was pressed in the terminal (non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()  # consume the newline
            break

    if len(clicked_pixels) < 4:
        print(f"\n  Need at least 4 points but only got {len(clicked_pixels)}.")
        print("  Re-run and click more points.")
        cv2.destroyAllWindows()
        return

    print(f"\n  Collected {len(clicked_pixels)} points. Moving to Phase 2.\n")

    # ── Phase 2: get world coordinates from terminal ──
    print("PHASE 2: Enter real-world (x, y) for each clicked point.\n")

    world_points = []
    for i, (px, py) in enumerate(clicked_pixels):
        while True:
            try:
                coords = input(f"  Point #{i+1} at pixel ({px}, {py}) — enter world x,y: ")
                parts = coords.strip().split(",")
                wx, wy = float(parts[0]), float(parts[1])
                world_points.append((wx, wy))
                break
            except (ValueError, IndexError):
                print("    Invalid. Enter as:  x,y  (e.g. 0.15,0.10)")

    # ── Compute homography ──
    src = np.array(clicked_pixels, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)
    H, status = cv2.findHomography(src, dst)

    if H is None:
        print("\nFailed to compute homography. Points may be collinear — try again.")
        cv2.destroyAllWindows()
        return

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "homography.npy")
    np.save(save_path, H)
    print(f"\nHomography saved to {save_path}")
    print(f"Matrix:\n{H}\n")

    # Verify reprojection
    print("Verification (reprojection error):")
    for i, (px, wy) in enumerate(zip(clicked_pixels, world_points)):
        pt = np.array([px[0], px[1], 1.0])
        reproj = H @ pt
        reproj /= reproj[2]
        err = np.linalg.norm(np.array(wy) - reproj[:2])
        print(f"  Point #{i+1}: expected=({wy[0]:.4f}, {wy[1]:.4f}), "
              f"got=({reproj[0]:.4f}, {reproj[1]:.4f}), error={err*1000:.1f}mm")

    print("\nDone. You can close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
