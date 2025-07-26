import os
import torch

# --------------------------- I/O Paths ---------------------------
RAW_DIR = '../raws/'
RESULTS_DIR = '../results/'

IMG1_PATH = os.path.join(RAW_DIR, 'set01_gradient1.pgm')
IMG2_PATH = os.path.join(RAW_DIR, 'set01_gradient2.pgm')
FACE_IMAGE_PATH = os.path.join(RAW_DIR, 'set01_image3.tiff')
REF_IMAGE_PATH = os.path.join(RAW_DIR, 'set01_image3.tiff')
NORMALS_PATH = os.path.join(RESULTS_DIR, 'normals.npy')
MASK_PATH = os.path.join(RESULTS_DIR, 'face_mask.npy')

# Output folders for different relighting styles
FACE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'relight_face')
SILVER_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'relight_silver')
GOLD_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'relight_gold')

# Default (for main relighting script)
GIF_PATH_FACE = os.path.join(FACE_OUTPUT_DIR, 'relight_face.gif')
GIF_PATH_SILVER = os.path.join(SILVER_OUTPUT_DIR, 'relight_silver.gif')
GIF_PATH_GOLD = os.path.join(GOLD_OUTPUT_DIR, 'relight_gold.gif')


# --------------------------- Semantic Segmentation ---------------------------
# Facial classes as defined by the SegFormer model
# FACE_CLASSES = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]

# for privacy purposes, we only use the following classes
FACE_CLASSES = [13, 17, 18]


# --------------------------- Device ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
