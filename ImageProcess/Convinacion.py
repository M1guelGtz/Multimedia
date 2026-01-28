from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
imagen = Image.open('ImageProcess/flores.png')  # Cambia la ruta a tu imagen
imagen2 = Image.open('ImageProcess/image.png')

imagen_gris = imagen.convert('L')
img_array = np.array(imagen_gris)
columnas, filas = imagen.size
umbral = 50
imagen_binaria = np.where(img_array > umbral, 255, 0).astype(np.uint8)
imagen_convinada = Image.new("RGB", (columnas, filas))

resultado = Image.fromarray(imagen_binaria)

columnas, filas = imagen.size
for i in range (columnas): 
    for j in range (filas):
        pixel = imagen_binaria[j, i]
        if pixel == 255:
            imagen_convinada.putpixel((i,j), imagen.getpixel((i,j)))
        
        else :
            imagen_convinada.putpixel((i,j), imagen2.getpixel((i,j)))
        
        
plt.figure(figsize=(15, 5))
plt.subplot(2, 3, 1)
plt.imshow(imagen)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(imagen2)
plt.title('Imagen de Fondo')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(imagen_gris, cmap='gray')
plt.title('Imagen en Escala de Grises')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(imagen_binaria, cmap='gray')
plt.title(f'Imagen Binarizada (umbral={umbral})')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(imagen_convinada, cmap='gray')
plt.title(f'Imagen Combinada')
plt.axis('off')

plt.tight_layout()
plt.show()