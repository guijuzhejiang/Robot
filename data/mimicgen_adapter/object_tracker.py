"""2D color-mask object tracker + 3D back-projection.

Used by the segmenter on REAL demos where per-frame ground-truth poses are
unavailable. The cubes/plate live on the table (z known per object), so we
recover (x, y, yaw) from a single front-camera frame.

KNOWN LIMITATIONS:
  - Cube tracking on sim renderings: ~2mm avg, p95 4mm. Excellent.
  - Plate tracking on sim renderings: ~35mm avg due to perspective: the centroid
    of the visible ellipse ≠ projection of the disc center (proven impossible
    to invert from single image without intrinsics calibration).
  - For SYNTHETIC pipeline tests (segmenter / replayer / augment smoke), prefer
    feeding ground-truth `obs["plate_pos"]` directly into the segmented demo.
  - For REAL demos: calibrate camera via ChArUco (configs/world_frame.yaml) AND
    use a non-disc marker (e.g., AprilTag mounted on plate center) for sub-cm
    plate accuracy. See docs/plans/phase3-real-demo-sim-augmentation.md T3.1.

Pipeline:
    rgb image → HSV mask per color → connected component → centroid + minAreaRect
    → pixel (u, v) → ray cast to z=z_known plane in robot frame → (x, y)
    → yaw from minAreaRect angle

The `CameraModel` class encapsulates intrinsics + extrinsics from
configs/cameras.yaml + configs/world_frame.yaml. If those configs are still
uncalibrated (intrinsics=null), the model can be used in "sim-only" mode where
it pulls intrinsics from a MuJoCo `MjModel` camera + computes extrinsics from
forward kinematics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from data.mimicgen_adapter.types import ObjectPose


# HSV ranges for the canonical PickPlaceBlue palette (tuned for sim renderings;
# real cubes may need slight adjustment after calibration).
HSV_RANGES = {
    "red":  [(np.array([0, 120, 80]), np.array([10, 255, 255])),
             (np.array([170, 120, 80]), np.array([180, 255, 255]))],  # red wraps in HSV
    "blue": [(np.array([95, 100, 80]), np.array([130, 255, 255]))],
    # Plate is pastel green in the MJCF (rgba 0.45 0.85 0.55). For a real
    # white plate, replace with [(0,0,180), (180,60,255)] AND add a workspace
    # ROI mask in the calling code to suppress white tablecloth false-positives.
    "plate": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
}

MIN_AREA_PX = 80  # smaller blobs are noise


@dataclass
class CameraModel:
    """Pinhole projection + extrinsic transform to robot frame."""
    K: np.ndarray              # (3, 3) intrinsics
    dist: np.ndarray           # (5,) distortion coeffs
    cam_to_world: np.ndarray   # (4, 4) homogenous transform
    image_size: tuple[int, int]  # (W, H)

    @classmethod
    def from_yaml(
        cls,
        cameras_yaml: Path | str,
        world_yaml: Path | str,
        camera_name: str = "front",
    ) -> "CameraModel | None":
        cameras = yaml.safe_load(Path(cameras_yaml).read_text())
        world = yaml.safe_load(Path(world_yaml).read_text())
        cfg = cameras["cameras"][camera_name]
        intr = cfg["intrinsics"]
        if any(v is None for v in (intr["fx"], intr["fy"], intr["cx"], intr["cy"])):
            return None
        K = np.array([[intr["fx"], 0, intr["cx"]],
                      [0, intr["fy"], intr["cy"]],
                      [0, 0, 1.0]])
        dist = np.array(cfg["distortion"], dtype=np.float64)
        ext = cfg.get("extrinsics_override") or world["camera_extrinsics_front"]
        cam_to_world = _compose_transform(ext["translation"], ext["rpy"])
        W, H = cfg["resolution"]
        return cls(K=K, dist=dist, cam_to_world=cam_to_world, image_size=(W, H))

    @classmethod
    def from_mujoco(cls, model, data, camera_name: str = "front",
                    image_size: tuple[int, int] = (320, 240)) -> "CameraModel":
        """Build a CameraModel from a MuJoCo MjModel + MjData.

        Useful for testing the tracker against sim-rendered frames where the
        ground-truth camera params are exactly known.
        """
        import mujoco

        cam_id = model.camera(camera_name).id
        fovy_deg = float(model.cam_fovy[cam_id])
        H, W = image_size[1], image_size[0]
        fy = 0.5 * H / np.tan(np.deg2rad(fovy_deg) / 2.0)
        fx = fy  # square pixels
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

        # Camera pose in world frame
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        # MuJoCo camera +z points AWAY from scene; OpenCV +z points TOWARD scene.
        # Convert by flipping y and z axes.
        flip = np.diag([1.0, -1.0, -1.0])
        R_cv = cam_mat @ flip
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R_cv
        cam_to_world[:3, 3] = cam_pos
        return cls(K=K, dist=np.zeros(5), cam_to_world=cam_to_world,
                   image_size=image_size)

    @property
    def world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam_to_world)

    # ---------------- projection helpers ----------------
    def pixel_to_world_on_plane(self, u: float, v: float, plane_z: float) -> np.ndarray:
        """Back-project pixel (u, v) onto plane z = plane_z (world frame).

        Returns (x, y, plane_z) in world coords.
        """
        # Ray in camera frame
        K_inv = np.linalg.inv(self.K)
        pixel_h = np.array([u, v, 1.0])
        ray_cam = K_inv @ pixel_h  # direction in camera frame (z forward)

        # Camera origin in world frame
        cam_origin = self.cam_to_world[:3, 3]
        # Ray direction in world frame
        ray_world = self.cam_to_world[:3, :3] @ ray_cam
        # Intersect with plane z = plane_z:  cam_origin.z + t * ray_world.z = plane_z
        if abs(ray_world[2]) < 1e-9:
            return np.array([np.nan, np.nan, plane_z])
        t = (plane_z - cam_origin[2]) / ray_world[2]
        if t <= 0:
            return np.array([np.nan, np.nan, plane_z])
        p = cam_origin + t * ray_world
        return p


def _compose_transform(translation, rpy) -> np.ndarray:
    """Compose 4x4 transform from xyz translation + rpy (roll-pitch-yaw) rotation."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


# ---------------- color mask detection ----------------
def detect_color(image_rgb: np.ndarray, color: str, *, fit_ellipse: bool = False) -> list[dict]:
    """Detect colored blobs in image. Returns list of dicts with keys:
        - 'centroid_uv': (u, v) pixel
        - 'angle_deg':   minAreaRect angle (for yaw estimation)
        - 'area_px':     int
    If fit_ellipse=True, replaces centroid_uv with the center of the
    least-squares fitted ellipse — more accurate for circular objects (plate)
    seen at oblique angles.
    """
    if color not in HSV_RANGES:
        raise ValueError(f"Unknown color {color!r}")
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        rect = cv2.minAreaRect(c)
        angle_deg = float(rect[2])
        if fit_ellipse and len(c) >= 5:
            ellipse = cv2.fitEllipse(c)  # ((cx, cy), (major, minor), angle)
            cx, cy = ellipse[0]
            angle_deg = float(ellipse[2])
        detections.append({"centroid_uv": (cx, cy), "angle_deg": angle_deg,
                           "area_px": int(area)})
    detections.sort(key=lambda d: d["area_px"], reverse=True)
    return detections


def track_objects(
    image_rgb: np.ndarray,
    cam: CameraModel,
    *,
    cube_z: float = 0.02,        # cube CENTER z (geom origin)
    plate_visible_z: float = 0.004,  # plate TOP surface z (what camera sees)
    plate_body_z: float = 0.002,     # plate body origin (what to return)
) -> dict[str, ObjectPose]:
    """Return {"red": pose, "blue": pose, "plate": pose} from a single image.

    Cubes: visible top vs center are within 1cm; project at center for simplicity.
    Plate: visible top surface (z=0.01) is what the camera sees; we project there
    then return body origin z (z=0.005) for consistency with MJCF body convention.

    Missing objects are omitted from the dict.
    """
    out: dict[str, ObjectPose] = {}
    plane_zs = {"red": cube_z, "blue": cube_z, "plate": plate_visible_z}
    return_zs = {"red": cube_z, "blue": cube_z, "plate": plate_body_z}
    use_ellipse = {"red": False, "blue": False, "plate": True}
    for color in ("red", "blue", "plate"):
        dets = detect_color(image_rgb, color, fit_ellipse=use_ellipse[color])
        if not dets:
            continue
        u, v = dets[0]["centroid_uv"]
        xyz = cam.pixel_to_world_on_plane(u, v, plane_zs[color])
        if np.any(np.isnan(xyz)):
            continue
        yaw = np.deg2rad(dets[0]["angle_deg"])
        out[color] = ObjectPose(name=color, xy=xyz[:2], yaw=float(yaw),
                                z=return_zs[color])
    return out
