import os
import cv2
import imageio
from PIL import Image

# ----------- Configuration -----------
INPUT_DIR = '../demo'               # 输入目录
OUTPUT_DIR = '../demo_resized'      # 输出目录
MAX_SIZE = 500                    # 最大边长度（长边压缩为此尺寸）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- Resize Utility (保持比例) -----------
def resize_image_keep_aspect(img, max_size=1000):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ----------- PNG 批量压缩 -----------
def resize_pngs():
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith('.png'):
            input_path = os.path.join(INPUT_DIR, fname)
            output_path = os.path.join(OUTPUT_DIR, fname)

            img = cv2.imread(input_path)
            resized = resize_image_keep_aspect(img, max_size=MAX_SIZE)
            cv2.imwrite(output_path, resized)
            print(f"[PNG] Resized {fname} → {resized.shape[1]}x{resized.shape[0]}")

# ----------- GIF 批量压缩 -----------
def resize_gifs():
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith('.gif'):
            input_path = os.path.join(INPUT_DIR, fname)
            output_path = os.path.join(OUTPUT_DIR, fname)

            reader = imageio.get_reader(input_path)
            meta = reader.get_meta_data()
            fps = meta.get('duration', 50) / 1000.0  # duration is in ms per frame

            frames = []
            for frame in reader:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for resizing
                resized = resize_image_keep_aspect(img, max_size=MAX_SIZE)
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(resized_rgb)

            imageio.mimsave(output_path, frames, duration=fps)
            print(f"[GIF] Resized {fname} → {len(frames)} frames")

# ----------- 主流程 -----------
if __name__ == "__main__":
    resize_pngs()
    resize_gifs()
    print("[DONE] All PNGs and GIFs resized.")
