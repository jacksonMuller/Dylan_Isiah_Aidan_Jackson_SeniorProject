"""
Convert 2D object detections (OpenCV DNN bounding boxes) into 3D target coordinates
in the robot base frame for inverse kinematics.

This intentionally uses a simple pinhole-camera model:
  - Use bbox center pixel (u, v)
  - Assume a depth Z_cam (distance from camera to the object)
  - Back-project to a 3D point in camera coordinates
  - Rotate/translate into the robot base frame

You will need to tune:
  - `depth_m` (or replace it with a real depth estimate)
  - `fov_deg_x` / `fov_deg_y`
  - `t_cam_to_base_m` (camera mount translation)
  - the camera->base rotation `R_cam_to_base`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fov(
    img_width_px: float,
    img_height_px: float,
    fov_deg_x: float,
    fov_deg_y: Optional[float] = None,
) -> CameraIntrinsics:
    """
    Approximate pinhole intrinsics from horizontal/vertical field-of-view.
    """
    if fov_deg_y is None:
        # Best-effort approximation for many webcams.
        fov_deg_y = fov_deg_x * (img_height_px / max(img_width_px, 1.0))

    fov_x_rad = math.radians(fov_deg_x)
    fov_y_rad = math.radians(float(fov_deg_y))

    fx = (img_width_px / 2.0) / math.tan(fov_x_rad / 2.0)
    fy = (img_height_px / 2.0) / math.tan(fov_y_rad / 2.0)

    cx = img_width_px / 2.0
    cy = img_height_px / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def bbox_center_px(bbox: Sequence[float]) -> Tuple[float, float]:
    """
    OpenCV DetectionModel returns bbox as [x, y, w, h].
    """
    if len(bbox) != 4:
        raise ValueError("Expected bbox format: [x, y, w, h]")
    x, y, w, h = bbox
    u = x + w / 2.0
    v = y + h / 2.0
    return float(u), float(v)


def pixel_to_camera_point_m(
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float, float]:
    """
    Back-project a pixel center (u, v) into a 3D point in camera coordinates.

    Camera coordinate convention used here:
      - x_cam: right
      - y_cam: down
      - z_cam: forward (out of the camera)
    """
    z_cam = float(depth_m)
    x_cam = (u_px - intrinsics.cx) / intrinsics.fx * z_cam
    y_cam = (v_px - intrinsics.cy) / intrinsics.fy * z_cam
    return float(x_cam), float(y_cam), float(z_cam)


def mat3_vec3_mul(
    R: Sequence[Sequence[float]],
    v: Sequence[float],
) -> Tuple[float, float, float]:
    """
    Multiply 3x3 matrix by a 3-vector.
    """
    if len(R) != 3 or any(len(row) != 3 for row in R):
        raise ValueError("R must be 3x3")
    if len(v) != 3:
        raise ValueError("v must be length 3")

    out = [0.0, 0.0, 0.0]
    for i in range(3):
        out[i] = R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]
    return out[0], out[1], out[2]


def compute_target_base_from_bbox(
    bbox_xywh: Sequence[float],
    img_width_px: float,
    img_height_px: float,
    *,
    depth_m: float,
    fov_deg_x: float = 60.0,
    fov_deg_y: Optional[float] = None,
    R_cam_to_base: Sequence[Sequence[float]] = (
        (0.0, 0.0, 1.0),   # base X = +cam Z
        (-1.0, 0.0, 0.0),  # base Y = -cam X
        (0.0, -1.0, 0.0),  # base Z = -cam Y
    ),
    t_cam_to_base_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> Dict[str, float]:
    """
    Convert bbox center pixel into a 3D target in robot base coordinates.

    Returns:
      {x, y, z} in base frame, plus {u, v, r, theta1}.
    """
    u_px, v_px = bbox_center_px(bbox_xywh)
    intr = intrinsics_from_fov(
        img_width_px=img_width_px,
        img_height_px=img_height_px,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
    )

    x_cam, y_cam, z_cam = pixel_to_camera_point_m(
        u_px=u_px,
        v_px=v_px,
        depth_m=depth_m,
        intrinsics=intr,
    )

    # base = R * cam + t
    x_b, y_b, z_b = mat3_vec3_mul(R_cam_to_base, (x_cam, y_cam, z_cam))
    tx, ty, tz = float(t_cam_to_base_m[0]), float(t_cam_to_base_m[1]), float(t_cam_to_base_m[2])
    x_b += tx
    y_b += ty
    z_b += tz

    r = math.hypot(x_b, y_b)
    theta1 = math.atan2(y_b, x_b)  # yaw angle about base Z

    return {
        "u": float(u_px),
        "v": float(v_px),
        "x": float(x_b),
        "y": float(y_b),
        "z": float(z_b),
        "r": float(r),
        "theta1_rad": float(theta1),
        "theta1_deg": float(math.degrees(theta1)),
    }


def pick_best_detection(
    object_infos: Iterable[Sequence],
    allowed_classes: Optional[Iterable[str]] = None,
) -> Optional[Tuple[Sequence[float], str, float]]:
    """
    Pick best detection by confidence.

    Expected item format: [bbox_xywh, class_name, confidence]
    (confidence may be missing; if so, returns the first match).
    """
    allowed = set(allowed_classes) if allowed_classes is not None else None

    best = None
    best_conf = -float("inf")
    for item in object_infos:
        if len(item) < 2:
            continue
        bbox = item[0]
        class_name = item[1]
        conf = float(item[2]) if len(item) >= 3 else 0.0

        if allowed is not None and class_name not in allowed:
            continue

        if conf > best_conf:
            best = (bbox, class_name, conf)
            best_conf = conf

    return best

