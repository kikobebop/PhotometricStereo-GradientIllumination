import numpy as np

def compute_normals_and_albedo(img1, img2, mask):
    """
    Compute surface normals and albedo from two gradient-illuminated RGB images.

    Parameters:
        img1 (np.ndarray): First gradient image (RGB, float32, [0, 1])
        img2 (np.ndarray): Second gradient image (RGB, float32, [0, 1])
        mask (np.ndarray): Binary face mask (H, W), values in {0, 1}

    Returns:
        normals (np.ndarray): Estimated surface normals, shape (H, W, 3), range [-1, 1]
        albedo (np.ndarray): Scalar albedo per pixel, shape (H, W), positive float
    """
    # Apply face mask
    masked_img1 = img1 * mask[:, :, None]
    masked_img2 = img2 * mask[:, :, None]

    # Compute normalized differences for x, y, z directions
    diff = masked_img1 - masked_img2
    summ = masked_img1 + masked_img2 + 1e-6  # to avoid division by zero

    nx = diff[:, :, 0] / summ[:, :, 0]
    ny = diff[:, :, 1] / summ[:, :, 1]
    nz = diff[:, :, 2] / summ[:, :, 2]

    normals = np.stack((nx, ny, nz), axis=2)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0] = 1.0  # avoid division by zero
    normals = normals / norm

    # Compute albedo from RGB gradient differences
    Gx = (2.0 / 3.0) * (masked_img1[:, :, 2] - masked_img2[:, :, 2])
    Gy = (2.0 / 3.0) * (masked_img1[:, :, 1] - masked_img2[:, :, 1])
    Gz = (2.0 / 3.0) * (masked_img1[:, :, 0] - masked_img2[:, :, 0])
    G = np.stack((Gx, Gy, Gz), axis=2)
    albedo = np.linalg.norm(G, axis=2)

    return normals, albedo
