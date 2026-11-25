"""nuScenes helpers: poses, point colouring and the virtual chase ("drone") camera."""
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
DYNAMIC = ("vehicle.", "human.", "animal", "movable_object.pushable_pullable")

# Chase cameras in the ego frame (x forward, y left, z up). "high": 13 m behind, 8.5 m up, looking 12 m ahead
# (the original). "low": 9 m behind, 4.5 m up, looking 15 m ahead and 0.5 m up — closer to the real cameras' ~1.5 m,
# so splat renders have fewer floaters.
CHASES = {
    "high": (np.array([-13.0, 0.0, 8.5]), np.array([12.0, 0.0, 0.0])),
    "low": (np.array([-9.0, 0.0, 4.5]), np.array([15.0, 0.0, 0.5])),
}
CHASE_POS, CHASE_LOOK = CHASES["high"]
CHASE_HFOV = 60.0


def tf(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def sensor_to_world(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return tf(pose["rotation"], pose["translation"]) @ tf(cs["rotation"], cs["translation"])


def ego_to_world(nusc, sd):
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return tf(pose["rotation"], pose["translation"])


def apply(T, pts):
    """pts: (N,3) -> (N,3)"""
    return pts @ T[:3, :3].T + T[:3, 3]


def chase_camera(W, H, chase="high"):
    pos, look = CHASES[chase]
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])  # rows: image x (right), image y (down), depth
    fx = (W / 2) / np.tan(np.radians(CHASE_HFOV / 2))
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    return R, K


def project_chase(pts_ego, W, H, chase="high"):
    R, K = chase_camera(W, H, chase)
    c = (pts_ego - CHASES[chase][0]) @ R.T
    z = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * c[:, 0] / z + K[0, 2]
        v = K[1, 1] * c[:, 1] / z + K[1, 2]
    return u, v, z, K[0, 0]


def splat(u, v, z, col, W, H, fx, radius_m=0.08, max_px=5):
    """Draw points as small squares, far to near. Returns RGBA uint8 (alpha = covered)."""
    ok = (z > 0.5) & np.isfinite(u) & (u > -max_px) & (u < W + max_px) & (v > -max_px) & (v < H + max_px)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]
    s = np.clip(np.round(radius_m * fx / z), 1, max_px).astype(int)
    half = s // 2
    pix, dep, cols = [], [], []
    for dx in range(-(max_px // 2), max_px // 2 + 1):
        for dy in range(-(max_px // 2), max_px // 2 + 1):
            m = (abs(dx) <= half) & (abs(dy) <= half)
            if not m.any():
                continue
            x = np.round(u[m]).astype(int) + dx
            y = np.round(v[m]).astype(int) + dy
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            pix.append(y[inb] * W + x[inb]); dep.append(z[m][inb]); cols.append(col[m][inb])
    img = np.zeros((H * W, 4), np.uint8)
    if pix:
        pix, dep, cols = np.concatenate(pix), np.concatenate(dep), np.concatenate(cols)
        o = np.argsort(-dep, kind="stable")
        img[pix[o], :3] = cols[o]
        img[pix[o], 3] = 255
    return img.reshape(H, W, 4)


def colour_points(nusc, sample, pts_world, dataroot, image_cache=None):
    """Give each world point the colour of the first camera (of this sample) that sees it.
    Returns (colours uint8 (N,3), seen bool (N,))."""
    N = len(pts_world)
    col = np.zeros((N, 3), np.uint8)
    seen = np.zeros(N, bool)
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"])
        T = np.linalg.inv(sensor_to_world(nusc, sd))
        idx = np.where(~seen)[0]
        c = apply(T, pts_world[idx])
        z = c[:, 2]
        front = z > 1.0
        idx, c, z = idx[front], c[front], z[front]
        uv = (c @ K.T)[:, :2] / z[:, None]
        key = sd["filename"]
        if image_cache is not None and key in image_cache:
            img = image_cache[key]
        else:
            img = np.asarray(Image.open(f"{dataroot}/{key}").convert("RGB"))
            if image_cache is not None:
                image_cache[key] = img
        h, w = img.shape[:2]
        x, y = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inb = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        col[idx[inb]] = img[y[inb], x[inb]]
        seen[idx[inb]] = True
    return col, seen


def dynamic_mask(nusc, sample, pts_world):
    """True for points inside any annotated box of a movable category in this sample."""
    from nuscenes.utils.geometry_utils import points_in_box
    m = np.zeros(len(pts_world), bool)
    for ann in sample["anns"]:
        a = nusc.get("sample_annotation", ann)
        if not a["category_name"].startswith(DYNAMIC):
            continue
        box = nusc.get_box(ann)
        box.wlh = box.wlh * 1.15  # a little margin
        m |= points_in_box(box, pts_world.T)
    return m
