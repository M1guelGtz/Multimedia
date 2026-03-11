import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

filtroSobelH = [[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]]

filtroSobelV = [[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]]

filtroMedia5x5 = [[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]]
factor_media = 25.0

def filtrar_borde(matriz, filtro, x, y):
    suma_total = 0
    offset = len(filtro) // 2
    for a in range(-offset, offset + 1):
        for b in range(-offset, offset + 1):
            pixel = matriz[x + a, y + b]
            peso = filtro[a + offset][b + offset]
            suma_total += pixel * peso
    return suma_total

def filtrar_media(matriz, filtro, x, y, factor):
    suma_total = 0
    offset = len(filtro) // 2
    for a in range(-offset, offset + 1):
        for b in range(-offset, offset + 1):
            pixel = matriz[x + a, y + b]
            peso = filtro[a + offset][b + offset]
            suma_total += pixel * peso
    return suma_total / factor

print("--- Procesando Imagen 1 (Radiografía con Sobel) ---")
img1_pil = Image.open('./Corte 2/Radiografia.jpg').convert('L')
entrada1 = np.array(img1_pil, dtype=float)
salida1 = np.zeros_like(entrada1)
filas1, columnas1 = entrada1.shape

start_time = time.time()
offset_sobel = len(filtroSobelH) // 2
for i in range(offset_sobel, filas1 - offset_sobel):
    for j in range(offset_sobel, columnas1 - offset_sobel):
        gx = filtrar_borde(entrada1, filtroSobelH, i, j)
        gy = filtrar_borde(entrada1, filtroSobelV, i, j)
        salida1[i, j] = np.sqrt(gx**2 + gy**2)
print(f"Terminado en {time.time() - start_time:.2f} segundos.")

salida1 = np.clip(salida1, 0, 255)
salida1_img = Image.fromarray(np.uint8(salida1))

print("\n--- Procesando Imagen 2 (Tornillo con Filtro Media 5x5) ---")
img2_pil = Image.open('./Corte 2/Tornillo_ruido2.png').convert('L')
entrada2 = np.array(img2_pil, dtype=float)
salida2 = np.zeros_like(entrada2)
filas2, columnas2 = entrada2.shape

start_time = time.time()
offset_media = len(filtroMedia5x5) // 2
for i in range(offset_media, filas2 - offset_media):
    for j in range(offset_media, columnas2 - offset_media):
        salida2[i, j] = filtrar_media(entrada2, filtroMedia5x5, i, j, factor_media)
print(f"Terminado en {time.time() - start_time:.2f} segundos.")

salida2 = np.clip(salida2, 0, 255)
salida2_img = Image.fromarray(np.uint8(salida2))

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.title("Original: Radiografía")
plt.imshow(img1_pil, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Procesada: Bordes Sobel")
plt.imshow(salida1_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Original: Tornillo con Ruido")
plt.imshow(img2_pil, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Procesada: Filtro de Media 5x5")
plt.imshow(salida2_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()