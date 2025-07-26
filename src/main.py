import os
import numpy as np
import cv2
from demosaic import load_and_demosaic
from align import align_images
from mask import get_face_mask
from normal import compute_normals_and_albedo
from config import IMG1_PATH, IMG2_PATH, FACE_IMAGE_PATH, RESULTS_DIR

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Step 1: Load and Demosaic Gradient Images ---
    gradient_img1 = load_and_demosaic(IMG1_PATH)
    gradient_img2 = load_and_demosaic(IMG2_PATH)

    # --- Step 2: Align Images using ORB + RANSAC ---
    aligned_img2 = align_images(gradient_img1, gradient_img2)

    # --- Step 3: Generate Face Mask using SegFormer ---
    face_mask = get_face_mask(FACE_IMAGE_PATH, gradient_img1.shape[:2])

    # --- Step 4: Compute Normals and Albedo ---
    normals, albedo = compute_normals_and_albedo(gradient_img1, aligned_img2, face_mask)

    # --- Step 5: Save Results ---
    np.save(os.path.join(RESULTS_DIR, 'normals.npy'), normals)
    np.save(os.path.join(RESULTS_DIR, 'albedo.npy'), albedo)
    np.save(os.path.join(RESULTS_DIR, 'face_mask.npy'), face_mask)

    # Save normal map for visualization
    normal_map = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(RESULTS_DIR, 'normal_map.png'), normal_map)

    # --- Step 6: Print Stats ---
    print("[INFO] Normal stats:")
    print("  Mean:", np.mean(normals, axis=(0, 1)))
    print("  Min:", np.min(normals), "Max:", np.max(normals))

    print("[INFO] Albedo stats:")
    print("  Min:", np.min(albedo), "Max:", np.max(albedo))
    print("  Shape:", albedo.shape)
