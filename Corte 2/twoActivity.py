import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_rgb = "image.png"
imagen_original = Image.open(image_rgb).convert("RGB")
rgb = np.array(imagen_original).astype(float)

YCbCr = [
    [0.299, 0.587, 0.114],
    [-0.169, -0.331, 0.500],
    [0.500, -0.419, -0.081]
]

r = rgb[:,:,0]
g = rgb[:,:,1]
b = rgb[:,:,2]

filas, columnas = r.shape

y  = ((YCbCr[0][0] * r) + YCbCr[0][1] * g + YCbCr[0][2] * b)  + 0
cb = ((YCbCr[1][0] * r) + YCbCr[1][1] * g + YCbCr[1][2] * b) + 128
cr = ((YCbCr[2][0] * r) + YCbCr[2][1] * g + YCbCr[2][2] * b) + 128

y = np.clip(y, 0, 255)
cb = np.clip(cb, 0, 255)
cr = np.clip(cr, 0, 255)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(imagen_original)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis("off")
axes[0, 1].imshow(y, cmap="gray")
axes[0, 1].set_title(" (y)")
axes[0, 1].axis("off")
axes[0, 2].imshow(cb, cmap="gray")
axes[0, 2].set_title("(Cb)")
axes[0, 2].axis("off")
axes[1, 0].imshow(cr, cmap="gray")
axes[1, 0].set_title("(Cr)")
axes[1, 0].axis("off")
axes[1, 1].axis("off")
axes[1, 2].axis("off") 

plt.tight_layout()
plt.show()