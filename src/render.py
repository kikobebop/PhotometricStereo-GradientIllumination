import numpy as np

def relight_with_phong(normals, img_ref, light_dir, ref_dot, mask,
                       gain=1.0, kd=1.0, ks=0.2, shininess=32,
                       ambient=0.0, material_color=None, view_dir=None):
    """
    General Blinn-Phong relighting function with optional ambient and material color.

    Args:
        normals (np.ndarray): (H, W, 3) surface normals.
        img_ref (np.ndarray): (H, W, 3) reference image.
        light_dir (np.ndarray): (3,) light direction.
        ref_dot (np.ndarray): (H, W) dot product with reference lighting.
        mask (np.ndarray): (H, W) binary mask.
        gain (float): Brightness multiplier.
        kd (float): Diffuse coefficient.
        ks (float): Specular coefficient.
        shininess (int): Shininess exponent.
        ambient (float): Ambient light coefficient. (0 = off)
        material_color (np.ndarray or None): (3,) RGB material tint for metal.
        view_dir (np.ndarray or None): (3,) view direction (default = [0, 0, 1])

    Returns:
        np.ndarray: Rendered RGB image (uint8).
    """
    H, W, _ = normals.shape

    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)
    view_dir = np.array([0, 0, 1], dtype=np.float32) if view_dir is None else view_dir
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-6)

    dot = np.clip(np.sum(normals * light_dir[None, None, :], axis=2), 1e-4, 1.0)
    stable_ref_dot = np.maximum(ref_dot, 5e-2)
    diff_ratio = np.clip(dot / stable_ref_dot, 0.0, 5.0)

    H_vec = (light_dir + view_dir)
    H_vec /= np.linalg.norm(H_vec)
    H_vec = H_vec[None, None, :]

    spec = np.clip(np.sum(normals * H_vec, axis=2), 0, 1.0)
    specular = (spec ** shininess)[..., None]

    if material_color is not None:
        # metallic material
        facing = np.clip(dot, 0, 1)
        boost = 1.0 + 2.0 * facing
        specular = ks * specular * material_color[None, None, :] * boost[..., None]
        ambient_term = ambient * material_color[None, None, :]
        diffuse_term = kd * img_ref * diff_ratio[..., None]
        shaded = ambient_term + diffuse_term + specular
    else:
        # default non-metallic
        shaded = kd * img_ref * diff_ratio[..., None] * gain + ks * specular

    shaded *= mask[..., None]
    return np.clip(shaded * 255.0, 0, 255).astype(np.uint8)
