from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

imagen_pil = Image.open('./Corte 2/img.jpeg').convert('RGB')
imagen_np = np.array(imagen_pil)
rgb = imagen_np.astype(float)

r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
umbral_diff = 60 
imagen_binaria = np.where((r - g > umbral_diff) & (r - b > umbral_diff), 255, 0).astype(np.uint8)
def dilatacion(img):
    res = np.zeros_like(img)
    for i, j in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        res = np.maximum(res, np.roll(np.roll(img, i, 0), j, 1))
    return res

def erosion(img):
    res = np.full_like(img, 255)
    for i, j in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        res = np.minimum(res, np.roll(np.roll(img, i, 0), j, 1))
    return res

imagenErosion = erosion(imagen_binaria)
imagenDilatacion = dilatacion(imagenErosion)

gradiente = dilatacion(imagenDilatacion).astype(np.int16) - erosion(imagenDilatacion).astype(np.int16)
gradiente = np.clip(gradiente, 0, 255).astype(np.uint8)

imagen_salida = imagen_np.copy()
color_marca = [0, 0, 0] 
imagen_salida[gradiente > 0] = color_marca

plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.imshow(imagen_np)
plt.title('1. Imagen Original', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(imagen_binaria, cmap='gray')
plt.title('2. Imagen Binarizada (Detección de Rojo)', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gradiente, cmap='gray')
plt.title('3. Gradiente Morfológico (Solo Bordes)', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(imagen_salida)
plt.title('4. Resultado: Original con Frutas Marcadas', fontsize=15)
plt.axis('off')

plt.tight_layout()
plt.show()