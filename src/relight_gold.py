import os
import cv2
import numpy as np
import imageio

from config import RESULTS_DIR, REF_IMAGE_PATH, NORMALS_PATH, MASK_PATH, GOLD_OUTPUT_DIR, GIF_PATH_GOLD
from light import estimate_light_direction, generate_equatorial_light_path
from render import relight_with_phong

GIF_PATH = GIF_PATH_GOLD
OUTPUT_DIR = GOLD_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Load Data ---------------------------
normals = np.load(NORMALS_PATH)  # (H, W, 3)
mask = np.load(MASK_PATH)        # (H, W)

ref_img_rgb = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
ref_img_rgb = ref_img_rgb[:, :, ::-1].astype(np.float32) / 65535.0

# --------------------------- Estimate Reference Light Direction ---------------------------
ref_light_dir = estimate_light_direction(normals, ref_img_rgb, mask)
ref_dot = np.sum(normals * ref_light_dir[None, None, :], axis=2)

# --------------------------- Render All Frames ---------------------------
print(f"[INFO] Generating metal-style relighting frames in {OUTPUT_DIR}...")
light_dirs = generate_equatorial_light_path(n_frames=60, n_turns=2)
frames = []
gold_rgb = np.array([1.00, 0.72, 0.06], dtype=np.float32)

for idx, light_dir in enumerate(light_dirs):
    print(f"Rendering frame {idx + 1}/{len(light_dirs)}")
    frame = relight_with_phong(normals, ref_img_rgb, light_dir, ref_dot, mask,
                               gain=1.0, kd=0.01, ks=1.6, shininess=64, ambient=0.01,
                               material_color=gold_rgb, view_dir=np.array([0, -1, 0]))
    out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}.png")
    cv2.imwrite(out_path, frame[:, :, ::-1])
    frames.append(frame)

print("[INFO] All frames rendered.")

# --------------------------- Save as GIF ---------------------------
imageio.mimsave(GIF_PATH, frames, duration=0.05)
print(f"[INFO] Saved relighting animation to: {GIF_PATH}")
