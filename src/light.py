import numpy as np

def estimate_light_direction(normals, img_ref, mask):
    """
    Estimate the light direction in the reference image using least squares.

    Args:
        normals (np.ndarray): Normal map, shape (H, W, 3)
        img_ref (np.ndarray): Reference image (H, W, 3)
        mask (np.ndarray): Binary face mask (H, W)

    Returns:
        np.ndarray: Estimated lighting direction (3,)
    """
    I = np.mean(img_ref, axis=2)[mask == 1]   # mean intensity in mask
    N = normals[mask == 1]                   # normal vectors in mask
    L, _, _, _ = np.linalg.lstsq(N, I, rcond=None)
    return L / (np.linalg.norm(L) + 1e-6)

def generate_equatorial_light_path(n_frames=60, n_turns=2):
    """
    Generate a list of light directions simulating equatorial circular motion.

    Args:
        n_frames (int): Number of light positions.
        n_turns (int): Number of full 360-degree turns.

    Returns:
        List[np.ndarray]: List of 3D light directions.
    """
    light_dirs = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        phi = 2 * np.pi * n_turns * t
        x = 0.0
        y = np.sin(phi)
        z = np.cos(phi)
        light_dirs.append(np.array([x, y, z]))
    return light_dirs