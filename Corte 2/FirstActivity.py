import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_rgb = "image.png"
imagen_original = Image.open(image_rgb).convert("RGB")
rgb = np.array(imagen_original).astype(float) / 255.0

r = rgb[:,:,0]
g = rgb[:,:,1]
b = rgb[:,:,2]

c = 1 - r
m = 1 - g
y = 1 - b

k = np.minimum(np.minimum(c, m), y)

filas, columnas = r.shape

vis_c = np.ones((filas, columnas, 3))
vis_c[:,:,0] = 1 - c

vis_m = np.ones((filas, columnas, 3))
vis_m[:,:,1] = 1 - m

vis_y = np.ones((filas, columnas, 3))
vis_y[:,:,2] = 1 - y

vis_k = np.ones((filas, columnas, 3))
vis_k[:,:,0] = 1 - k
vis_k[:,:,1] = 1 - k
vis_k[:,:,2] = 1 - k

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis("off")
axes[0, 1].imshow(vis_c)
axes[0, 1].set_title("Cian (C)")
axes[0, 1].axis("off")
axes[0, 2].imshow(vis_m)
axes[0, 2].set_title("Magenta (M)")
axes[0, 2].axis("off")
axes[1, 0].imshow(vis_y)
axes[1, 0].set_title("Yellow (Y)")
axes[1, 0].axis("off")
axes[1, 1].imshow(vis_k)
axes[1, 1].set_title("Black (K)")
axes[1, 1].axis("off")
axes[1, 2].axis("off") 

plt.tight_layout()
plt.show()