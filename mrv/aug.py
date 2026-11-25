"""Phone-photo augmentation shared by real (scripts/encode_aug.py) and sim (scripts/sim_encode.py) encoders."""
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pyquaternion import Quaternion

u0 = (np.arange(16) + 0.5) / 16
v0 = (np.arange(9) + 0.5) / 9
UU, VV = np.meshgrid(u0, v0)


def augment(img, rng):
    W, H = img.size
    z = rng.uniform(1.0, 1.3)
    cw, ch = W / z, H / z
    ox, oy = rng.uniform(0, W - cw), rng.uniform(0, H - ch)
    im = img.crop((ox, oy, ox + cw, oy + ch)).resize((448, 252), Image.BICUBIC)
    for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = E(im).enhance(rng.uniform(0.75, 1.25))
    x = np.asarray(im).astype(np.float32) / 255
    x = x ** rng.uniform(0.8, 1.25)
    hsv = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-0.03, 0.03) * 255)) % 256
    im = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(rng.uniform(35, 95))); buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB")), (ox, oy, cw, ch)


def rays_for(cs, box):
    K = np.array(cs["camera_intrinsic"], dtype=np.float64)
    R = Quaternion(cs["rotation"]).rotation_matrix
    o = np.array(cs["translation"])
    ox, oy, cw, ch = box
    pix = np.stack([ox + UU.ravel() * cw, oy + VV.ravel() * ch, np.ones(144)], 1)  # original 1600x900 pixels
    d = (np.linalg.inv(K) @ pix.T).T @ R.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([d, np.cross(np.broadcast_to(o, d.shape), d)], 1)
