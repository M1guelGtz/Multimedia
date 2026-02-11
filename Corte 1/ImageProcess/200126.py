from PIL import Image
import matplotlib.pyplot as plt 

image_lena = Image.open("image.png")


columnas, filas = image_lena.size
imagen_blanco = Image.new("L", (columnas, filas))
imagen_invertida = Image.new("RGB", (columnas, filas))
imagen_invertida_bc = Image.new("L", (columnas, filas))
imagen_negativa = Image.new("RGB", (columnas, filas))
imagen_aclarada = Image.new("RGB", (columnas, filas))


for i in range (columnas): 
    for j in range (filas):
        #obtener el pixel en la posicion ( i, j )
        red, green, blue = image_lena.getpixel( (i,j) )
        # Invertir lo colores
        imagen_negativa.putpixel((i,j), (255 - red, 255 - green, 255 - blue))
        #Aclarar la Imagen
        imagen_aclarada.putpixel((i,j), (red+100, green + 100, blue + 100))
        #Calcular el tono de gris
        gray_pixel = int ((red + green + blue) / 3)
        # Asignar tono de gris
        imagen_blanco.putpixel((i,j), gray_pixel)
        # Invertir la imagen (espejo horizontal)
        imagen_invertida.putpixel((i,j), image_lena.getpixel((columnas - i - 1, j)))
        
# Invertir la imagen (espejo horizontal)

# Primera Imagen        
plt.subplot(1,5,1)
plt.imshow(image_lena, cmap='gray')
plt.title("Original")
plt.axis("off")

#Segunda Imagen
plt.subplot(1,5,2)
plt.imshow(imagen_blanco, cmap='gray')
plt.title("Escala de Grises")
plt.axis("off")


plt.subplot(1,5,3)
plt.imshow(imagen_invertida, cmap='gray')
plt.title("Invertida")
plt.axis("off")


plt.subplot(1,5,4)
plt.imshow(imagen_negativa)
plt.title("Negativa")
plt.axis("off")

# aclrarar la imagen, sacarle negativo e invertir la imagen
plt.subplot(1,5,5)
plt.imshow(imagen_aclarada)
plt.title("Aclarada")
plt.axis("off")
plt.show()