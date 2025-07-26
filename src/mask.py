import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from config import device, FACE_CLASSES


def get_face_mask(img_path, target_shape):
    """
    Generate a binary face mask using SegFormer segmentation model.

    Parameters:
        img_path (str): Path to RGB face image
        target_shape (tuple): Target (height, width) to resize mask

    Returns:
        np.ndarray: Binary face mask of shape target_shape (values 0 or 1)
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.dtype != np.uint8:
        img = ((img.astype(np.float32) / 65535) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device)

    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
    labels = up.argmax(1)[0].cpu().numpy()

    mask = np.isin(labels, FACE_CLASSES).astype(np.uint8)
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
