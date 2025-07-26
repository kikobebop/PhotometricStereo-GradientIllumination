import os
import cv2
import numpy as np
import imageio

from config import RESULTS_DIR, REF_IMAGE_PATH, NORMALS_PATH, MASK_PATH, FACE_OUTPUT_DIR, GIF_PATH_FACE
from light import estimate_light_direction, generate_equatorial_light_path
from render import relight_with_phong

GIF_PATH = GIF_PATH_FACE
OUTPUT_DIR = FACE_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Load Data ---------------------------
normals = np.load(NORMALS_PATH)  # (H, W, 3)
mask = np.load(MASK_PATH)        # (H, W), uint8

ref_img_rgb = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
ref_img_rgb = ref_img_rgb[:, :, ::-1].astype(np.float32) / 65535.0  # Convert BGR to RGB

# --------------------------- Estimate Reference Light Direction ---------------------------
ref_light_dir = estimate_light_direction(normals, ref_img_rgb, mask)
ref_dot = np.sum(normals * ref_light_dir[None, None, :], axis=2)

# --------------------------- Generate Relighting Frames ---------------------------
print(f"[INFO] Generating relighting frames in {OUTPUT_DIR}...")
light_dirs = generate_equatorial_light_path(n_frames=60, n_turns=2)
frames = []

for idx, light_dir in enumerate(light_dirs):
    print(f"Rendering frame {idx + 1}/{len(light_dirs)}")
    frame = relight_with_phong(normals, ref_img_rgb, light_dir, ref_dot, mask,
                               gain=1.0, kd=1.0, ks=0.15, shininess=96)
    out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}.png")
    cv2.imwrite(out_path, frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
    frames.append(frame)

print("[INFO] All frames rendered.")

# --------------------------- Save as GIF ---------------------------
imageio.mimsave(GIF_PATH, frames, duration=0.05)
print(f"[INFO] Saved face relighting animation to: {GIF_PATH}")