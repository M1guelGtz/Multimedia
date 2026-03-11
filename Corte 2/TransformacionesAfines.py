# --- IMPORTACIONES (De la segunda imagen) ---
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math
import numpy as np
from math import cos, sin, tan, radians
# Cargar imagen en escala de grises
imagen = Image.open('image.png').convert("L")

# Obtener dimensiones correctamente
columna, renglon = imagen.size

# Crear nueva imagen con fondo blanco
imagen_resultante = Image.new('L', (columna, renglon), 'white')

angulo = 30
theta = radians(angulo)

# Matriz de traslación (desplaza 80 píxeles en X y 40 en Y)
matriz_traslacion = np.array([[cos(theta), -cos(theta), 0],
                              [sin(theta), cos(theta), 0],
                              [0, 0, 1]])

# Recorrer los píxeles de la imagen original
for i in range(renglon):
    for j in range(columna):
        valor_pixel = imagen.getpixel((j, i))  # Acceder en orden correcto (columna, fila)

        # Crear coordenada homogénea
        coordenada = np.array([[j], [i], [1]])

        # Aplicar transformación
        nueva_coordenada = matriz_traslacion @ coordenada

        # Obtener nuevos valores de coordenadas
        x_nuevo, y_nuevo = int(nueva_coordenada[0, 0]), int(nueva_coordenada[1, 0])

        # Verificar que la nueva coordenada esté dentro de los límites de la imagen
        if 0 <= x_nuevo < columna and 0 <= y_nuevo < renglon:
            imagen_resultante.putpixel((x_nuevo, y_nuevo), valor_pixel)

# # Aplicar filtro de mediana
# imagen_filtrada = imagen_resultante.filter(ImageFilter.MedianFilter(size=3))

# Mostrar la imagen resultante
imagen_resultante