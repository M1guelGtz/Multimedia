from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ruta_msg1 = "./ImageProcess/Actividad9/mensaje_oculto1.png"
ruta_msg2 = "./ImageProcess/Actividad9/mensaje_oculto2.png"
img_msg1 = Image.open(ruta_msg1).convert("L")
img_msg2 = Image.open(ruta_msg2).convert("L")
matriz_img1 = np.array(img_msg1, dtype=np.int32)
matriz_img2 = np.array(img_msg2, dtype=np.int32)
matriz_resta = np.abs(matriz_img1 - matriz_img2)
matriz_resta = np.uint8(matriz_resta)
imagen_descifrada = Image.fromarray(matriz_resta, mode="L")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_msg1, cmap='gray')
axes[0].set_title("Imagen Original 1")
axes[0].axis("off")
axes[1].imshow(img_msg2, cmap='gray')
axes[1].set_title("Imagen Original 2")
axes[1].axis("off")
axes[2].imshow(imagen_descifrada, cmap='gray')
axes[2].set_title("Resta: |Img1 - Img2|\n(Mensaje Descifrado)")
axes[2].axis("off")
umbral = 127
imagen_binarizada = np.where(matriz_resta > umbral, 255, 0).astype(np.uint8)
imagen_binarizada_pil = Image.fromarray(imagen_binarizada, mode="L")
axes[3].imshow(imagen_binarizada_pil, cmap='gray')
axes[3].set_title("Binarizada (umbral=127)")
axes[3].axis("off")
plt.tight_layout()
plt.savefig("./ImageProcess/Actividad9/01_Mensaje_Descifrado.png", dpi=150, bbox_inches='tight')
plt.show()
imagen_descifrada.save("./ImageProcess/Actividad9/mensaje_descifrado_resultado.png")
archivos = [
    "./ImageProcess/Actividad9/Imagen_con_detalles_escondidos1.png",
    "./ImageProcess/Actividad9/Imagen_con_detalles_escondidos2.png",
    "./ImageProcess/Actividad9/Imagen_con_detalles_escondidos3.png",
    "./ImageProcess/Actividad9/Imagen_con_detalles_escondidos4.png",
    "./ImageProcess/Actividad9/Imagen_con_detalles_escondidos5.png"
]
imagenes_cargadas = []
for archivo in archivos:
    img = Image.open(archivo).convert('L')
    imagenes_cargadas.append(img)

columnas, filas = imagenes_cargadas[0].size

imagenes_binarizadas = []
for _ in range(5):
    imagenes_binarizadas.append(Image.new("L", (columnas, filas), 0))

imagen_suma_final = Image.new("L", (columnas, filas), 0)

UMBRAL_INFERIOR = 1
UMBRAL_SUPERIOR = 15
for i in range(5):
    img_actual = imagenes_cargadas[i]
    img_binaria = imagenes_binarizadas[i]
    for x in range(columnas):
        for y in range(filas):
            pixel_gris = img_actual.getpixel((x, y))
            
            if UMBRAL_INFERIOR <= pixel_gris <= UMBRAL_SUPERIOR:
                img_binaria.putpixel((x, y), 255)
            else:
                img_binaria.putpixel((x, y), 0)
for x in range(columnas):
    for y in range(filas):
        suma_pixeles = 0
        
        for i in range(5):
            suma_pixeles += imagenes_binarizadas[i].getpixel((x, y))
        
        valor_final = min(suma_pixeles, 255)
        imagen_suma_final.putpixel((x, y), valor_final)
plt.figure(figsize=(15, 10))
nombres_posiciones = ['Abajo-Derecha', 'Abajo-Izquierda', 'Arriba-Izquierda', 
                      'Arriba-Derecha', 'Centro']
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(imagenes_binarizadas[i], cmap='gray', vmin=0, vmax=255)
    plt.title(f'Figura {i+1}: {nombres_posiciones[i]}')
    plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(imagen_suma_final, cmap='gray', vmin=0, vmax=255)
plt.title('Resultado Final: Suma de las 5 Figuras Ocultas')
plt.axis('off')
plt.tight_layout()
plt.savefig("./ImageProcess/Actividad9/02_Figuras_y_Suma_Final.png", dpi=150, bbox_inches='tight')
plt.show()