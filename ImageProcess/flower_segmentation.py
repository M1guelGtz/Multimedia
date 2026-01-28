from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ruta_imagen = "./ImageProcess/flowers.jpg"
imagen_original = Image.open(ruta_imagen).convert("RGB")
img_array = np.array(imagen_original)
red_channel = img_array[:, :, 0]
green_channel = img_array[:, :, 1]
blue_channel = img_array[:, :, 2]
print("\nUMBRALES")
print("Criterios para segmentar SOLO la flor roja:")
print("- Canal Rojo > 200 (rojo muy intenso)")
print("- Canal Verde < 80 (excluye completamente amarillo)")
print("- Canal Azul < 100 (excluye fondo azul)")
print("- Rojo > Verde * 2.0 (dominancia fuerte del rojo)")
mascara_roja = np.zeros(red_channel.shape, dtype=np.uint8)
for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        r = red_channel[i, j]
        g = green_channel[i, j]
        b = blue_channel[i, j]
        if r > 150 and g < 5 and b < 150 and r > g * 2.0:
            mascara_roja[i, j] = 255
imagen_segmentada_color = Image.new("RGB", imagen_original.size, (0, 0, 0))
for i in range(img_array.shape[1]):
    for j in range(img_array.shape[0]):
        if mascara_roja[j, i] == 255:
            imagen_segmentada_color.putpixel((i, j), tuple(img_array[j, i]))
            
pixeles_flor_roja = np.sum(mascara_roja == 255)
area_total = mascara_roja.shape[0] * mascara_roja.shape[1]
porcentaje_flor = (pixeles_flor_roja / area_total) * 100

print(f"\nRedultado")
print(f"flor roja segmentada: {pixeles_flor_roja}")
print(f"imagen: {porcentaje_flor:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen_original)
plt.title("Imagen Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(imagen_segmentada_color)
plt.title("Flor Roja Segmentada")
plt.axis("off")
plt.tight_layout()
plt.show()