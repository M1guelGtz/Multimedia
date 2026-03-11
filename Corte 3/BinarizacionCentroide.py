from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

imagen_original = Image.open('./Corte 3/seagull.jpg').convert("RGB")
imagen_gris = imagen_original.convert("L")
columnas, filas = imagen_gris.size
umbral = 30
imagen_binaria = Image.new("L", (columnas, filas))

for i in range(columnas):
    for j in range(filas):
        pixel = imagen_gris.getpixel((i, j))
        if pixel <= umbral:
            imagen_binaria.putpixel((i, j), 255)  # Águila en blanco
        else:
            imagen_binaria.putpixel((i, j), 0)    # Fondo en negro

array_binario = np.array(imagen_binaria)
M00 = 0
M10 = 0 
M01 = 0 
for y in range(filas):
    for x in range(columnas):
        intensidad = int(array_binario[y, x])
        M00 += intensidad
        M10 += x * intensidad        
        M01 += y * intensidad

print(f"\nMomentos del águila:")
print(f"M00 = {M00:.0f}")
print(f"M10 = {M10:.0f}")
print(f"M01 = {M01:.0f}")
if M00 != 0:
    x_centroide = M10 / M00
    y_centroide = M01 / M00
    print(f"\nCentroide del águila:")
    print(f"x = M10/M00 = {M10:.0f}/{M00:.0f} = {x_centroide:.2f}")
    print(f"y = M01/M00 = {M01:.0f}/{M00:.0f} = {y_centroide:.2f}")
else:
    print("\nError: M00 es cero, no se puede calcular el centroide")
    x_centroide = y_centroide = 0
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagen_original)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imagen_gris, cmap='gray')
plt.title("Escala de Grises")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imagen_binaria, cmap='gray')
plt.title(f"Imagen Binarizada (Umbral={umbral})")

if M00 != 0:
    plt.plot(x_centroide, y_centroide, 'r+', markersize=20, markeredgewidth=3)
    plt.plot(x_centroide, y_centroide, 'ro', markersize=10, fillstyle='none', markeredgewidth=2)
    plt.text(x_centroide + 10, y_centroide + 10, 
             f'Centroide\n({x_centroide:.1f}, {y_centroide:.1f})', 
             color='red', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.axis("off")

plt.tight_layout()
plt.show()