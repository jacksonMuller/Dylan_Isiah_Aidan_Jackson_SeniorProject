import cv2
import numpy as np
import mujoco


def capture_frame_sim(m, d, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render a frame from a MuJoCo camera. Returns a BGR image (OpenCV format)."""
    renderer = mujoco.Renderer(m, height=height, width=width)
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=camera_name)
    rgb = renderer.render()
    renderer.close()
    # MuJoCo gives RGB top-down; OpenCV uses BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def capture_frame_real(cap: cv2.VideoCapture) -> np.ndarray:
    """Read a frame from a USB webcam. Returns BGR image or None on failure."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def detect_object(frame: np.ndarray, config: dict) -> tuple:
    """
    Detect the largest colored object in the frame using HSV thresholding.

    Returns (cx, cy) pixel center of the detection, or None if nothing found.
    """
    det = config["detection"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array(det["hsv_lower"], dtype=np.uint8)
    upper1 = np.array(det["hsv_upper"], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Second range (for colors like red that wrap around HSV)
    lower2 = np.array(det["hsv_lower2"], dtype=np.uint8)
    upper2 = np.array(det["hsv_upper2"], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 | mask2

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find largest contour above minimum area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < det["min_area"]:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def pixel_to_world(pixel_xy: tuple, homography: np.ndarray) -> tuple:
    """
    Convert a pixel coordinate to real-world (x, y) using a 3x3 homography matrix.

    The homography maps pixel (u, v) -> world (x, y) on the workspace plane.
    """
    pt = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    world_h = homography @ pt
    # Normalize homogeneous coordinates
    wx = world_h[0] / world_h[2]
    wy = world_h[1] / world_h[2]
    return (float(wx), float(wy))


def draw_detection(frame: np.ndarray, pixel_xy: tuple, world_xy: tuple = None) -> np.ndarray:
    """Draw a crosshair and optional world coordinates on the frame for debugging."""
    annotated = frame.copy()
    cx, cy = pixel_xy
    cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    label = f"px:({cx},{cy})"
    if world_xy is not None:
        label += f" world:({world_xy[0]:.3f},{world_xy[1]:.3f})"
    cv2.putText(annotated, label, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated
