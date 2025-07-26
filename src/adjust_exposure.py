import imageio.v2 as imageio
import numpy as np

# 读入两张 pgm 图（线性空间）
img1 = imageio.imread("../raws/set02_gradient1.pgm").astype(np.float32)
img2 = imageio.imread("../raws/set02_gradient2.pgm").astype(np.float32)

# 计算亮度缩放因子（例如使用平均值或中位数）
scale = np.median(img2) / (np.median(img1) + 1e-6)

# 对 img1 增益
corrected_img1 = np.clip(img1 * scale, 0, 65535).astype(np.uint16)

# 保存
imageio.imwrite("../raws/set02_gradient1_corrected.pgm", corrected_img1)
