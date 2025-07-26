import cv2
import numpy as np

def align_images(img1, img2):
    """
    Align img2 to img1 using ORB feature matching and RANSAC homography.

    Parameters:
        img1 (np.ndarray): Reference image
        img2 (np.ndarray): Image to align

    Returns:
        np.ndarray: img2 warped into the coordinate frame of img1
    """
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    print(f"[INFO] ORB matches found: {len(matches)}")

    if len(matches) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        print("[INFO] Alignment successful.")
        return aligned
    else:
        print("[WARN] Not enough matches. Skipping alignment.")
        return img2.copy()