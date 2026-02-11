from PIL import Image
import matplotlib.pyplot as plt

ruta_imagen = "./ImageProcess/pen.jpeg"
imagen_original = Image.open(ruta_imagen).convert("RGB")
imagen_grises = imagen_original.convert("L")

histograma_grises = imagen_grises.histogram()
total_pixeles = imagen_grises.width * imagen_grises.height
suma_ponderada_total = sum(i * histograma_grises[i] for i in range(256))

suma_fondo = 0
peso_fondo = 0
varianza_maxima = 0
umbral_otsu = 0

for t in range(256):
    peso_fondo += histograma_grises[t]
    if peso_fondo == 0: continue

    peso_objeto = total_pixeles - peso_fondo
    if peso_objeto == 0: break

    suma_fondo += t * histograma_grises[t]

    media_fondo = suma_fondo / peso_fondo
    media_objeto = (suma_ponderada_total - suma_fondo) / peso_objeto

    varianza = peso_fondo * peso_objeto * (media_fondo - media_objeto) ** 2

    if varianza > varianza_maxima:
        varianza_maxima = varianza
        umbral_otsu = t
imagen_binaria = imagen_grises.point(lambda p: 255 if p < umbral_otsu else 0)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagen_original)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.bar(range(256), histograma_grises, width=1.0, color='gray')
plt.axvline(umbral_otsu, color='red', linestyle="--", linewidth=2, label=f"Umbral: {umbral_otsu}")
plt.title("Histograma")
plt.xlabel("Nivel de gris")
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(imagen_binaria, cmap="gray")
plt.title("Imagen Binarizada")
plt.axis("off")

plt.tight_layout()
plt.show()