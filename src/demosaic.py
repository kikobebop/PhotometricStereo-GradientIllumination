# src/demosaic.py
import cv2
import numpy as np

def load_and_demosaic(path):
    """
    Load a Bayer-patterned 16-bit image from the given path
    and convert it to RGB format.

    Parameters:
        path (str): Path to the raw .pgm or .tiff image

    Returns:
        np.ndarray: Normalized RGB image in float32, range [0, 1]
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    color = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
    return color.astype(np.float32) / 65535.0
