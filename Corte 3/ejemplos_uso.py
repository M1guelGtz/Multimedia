"""
EJEMPLOS Y CASOS DE USO - Cambio de Fondo de Video con Chroma Key

Este archivo contiene ejemplos prácticos de cómo usar el procesador de video
con diferentes técnicas de procesamiento de imágenes.

REQUISITOS:
- opencv-python (cv2)
- numpy
- matplotlib
- pillow (PIL)
"""

import cv2
import numpy as np
from QuitarFondoVideo import ChromaKeyProcessor

# ============================================================================
# EJEMPLO 1: USO BÁSICO - Fondo sólido
# ============================================================================

def ejemplo_1_fondo_solido():
    """
    Ejemplo básico: Reemplazar fondo verde con color azul
    """
    print("EJEMPLO 1: Fondo Sólido")
    print("-" * 50)
    
    # Crear procesador
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    
    # Procesar video
    processor.procesar_video(
        metodo='basico',
        output_path='resultado_azul.mp4',
        fondo_tipo='color',
        fondo_valor=(255, 0, 0)  # BGR: Azul
    )
    print("✓ Video guardado: resultado_azul.mp4\n")


# ============================================================================
# EJEMPLO 2: Fondo Gradiente
# ============================================================================

def ejemplo_2_fondo_gradiente():
    """
    Crear un fondo con gradiente de color
    """
    print("EJEMPLO 2: Fondo Gradiente")
    print("-" * 50)
    
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    
    processor.procesar_video(
        metodo='avanzado',
        output_path='resultado_gradiente.mp4',
        fondo_tipo='gradiente'
    )
    print("✓ Video guardado: resultado_gradiente.mp4\n")


# ============================================================================
# EJEMPLO 3: Fondo Desenfocado (bokeh)
# ============================================================================

def ejemplo_3_fondo_desenfocado():
    """
    Crea un efecto bokeh desenfocando el frame original como fondo
    """
    print("EJEMPLO 3: Fondo Desenfocado (Bokeh)")
    print("-" * 50)
    
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    
    processor.procesar_video(
        metodo='basico',
        output_path='resultado_bokeh.mp4',
        fondo_tipo='desenfoque'
    )
    print("✓ Video guardado: resultado_bokeh.mp4\n")


# ============================================================================
# EJEMPLO 4: Fondo con Imagen
# ============================================================================

def ejemplo_4_fondo_imagen():
    """
    Usa una imagen como fondo
    """
    print("EJEMPLO 4: Fondo con Imagen")
    print("-" * 50)
    
    processor = ChromaKeyProcessor(
        video_path="video_chroma_key.mp4",
        fondo_path="mi_fondo.jpg"  # Cambia esto a tu imagen
    )
    
    processor.procesar_video(
        metodo='avanzado',
        output_path='resultado_fondo_imagen.mp4',
        fondo_tipo='imagen'
    )
    print("✓ Video guardado: resultado_fondo_imagen.mp4\n")


# ============================================================================
# EJEMPLO 5: Comparación Visual de Técnicas
# ============================================================================

def ejemplo_5_comparacion():
    """
    Muestra una comparación visual de todas las técnicas en un solo frame
    
    Técnicas aplicadas:
    1. Segmentación HSV - Detecta el verde en espacio HSV
    2. Segmentación RGB - Usa canales RGB directamente
    3. Procesamiento Básico - Segmentación + Apertura/Cierre
    4. Procesamiento Avanzado - Segmentación + Múltiples Op. Morfológicas
    5. Detección de Bordes Sobel - Identifica los bordes
    6. Detección de Bordes Laplaciano - Bordes más sensibles
    7. Resultado Básico - Composición final (método básico)
    8. Resultado Avanzado - Composición final (método avanzado)
    """
    print("EJEMPLO 5: Comparación Visual de Técnicas")
    print("-" * 50)
    
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    processor.mostrar_comparacion_frame(num_frame=50)  # Muestra frame 50
    print("✓ Comparación mostrada en pantalla\n")


# ============================================================================
# EJEMPLO 6: Procesamiento Personalizado
# ============================================================================

def ejemplo_6_personalizado():
    """
    Ejemplo de procesamiento personalizado con múltiples frames
    """
    print("EJEMPLO 6: Procesamiento Personalizado")
    print("-" * 50)
    
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    
    # Crear fondos personalizados
    fondo_rojo = np.zeros((processor.height, processor.width, 3), dtype=np.uint8)
    fondo_rojo[:] = (0, 0, 255)  # BGR: Rojo
    
    # Guardar fondos
    processor.fondo = fondo_rojo
    processor.procesar_video(
        metodo='avanzado',
        output_path='resultado_rojo.mp4',
        fondo_tipo='color',
        fondo_valor=(0, 0, 255)  # BGR: Rojo
    )
    print("✓ Video guardado: resultado_rojo.mp4\n")


# ============================================================================
# EJEMPLO 7: Explicación de Técnicas Utilizadas
# ============================================================================

def explicar_tecnicas():
    """
    Explicación de cada técnica implementada
    """
    print("""
    TÉCNICAS DE PROCESAMIENTO DE IMÁGENES UTILIZADAS
    ================================================
    
    1. SEGMENTACIÓN POR COLOR - HSV
       Descripción: Convierte el frame a espacio de color HSV (Hue, Saturation, Value)
       Ventaja: Más robusto a variaciones de iluminación
       Cómo funciona: Define rangos de HSV que correspondan al verde chroma key
       Código: cv2.inRange() con límites HSV
    
    2. SEGMENTACIÓN POR COLOR - RGB
       Descripción: Usa los canales RGB directamente
       Ventaja: Más simple de entender
       Cómo funciona: Aplica reglas tipo "G > R y G > B"
       Código: Operaciones con np.where()
    
    3. EROSIÓN
       Descripción: Reduce el tamaño de objetos blancos en la imagen
       Ventaja: Elimina ruido pequeño
       Cómo funciona: Expande píxeles negros hacia píxeles blancos
       Código: cv2.erode()
    
    4. DILATACIÓN
       Descripción: Expande el tamaño de objetos blancos
       Ventaja: Rellena agujeros pequeños
       Cómo funciona: Expande píxeles blancos hacia píxeles negros
       Código: cv2.dilate()
    
    5. APERTURA (Erosión + Dilatación)
       Descripción: Elimina ruido pequeño sin cambiar el tamaño de objetos grandes
       Ventaja: Limpieza suave de máscaras
       Código: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    6. CIERRE (Dilatación + Erosión)
       Descripción: Rellena agujeros pequeños dentro de objetos
       Ventaja: Mejora la cohesión del objeto
       Código: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    7. SUAVIZADO GAUSSIANO
       Descripción: Aplica un filtro Gaussiano
       Ventaja: Suaviza bordes para transiciones naturales
       Cómo funciona: Promedia píxeles vecinos con pesos Gaussianos
       Código: cv2.GaussianBlur()
    
    8. FILTRO SOBEL
       Descripción: Detecta bordes usando derivadas
       Ventaja: Identifica transiciones de intensidad
       Cómo funciona: Aplica dos kernels 3x3 (horizontal y vertical)
       Código: cv2.Sobel()
    
    9. FILTRO LAPLACIANO
       Descripción: Operador de segunda derivada
       Ventaja: Detección más sensible de bordes
       Cómo funciona: Detecta cambios abruptos en gradiente
       Código: cv2.Laplacian()
    
    10. COMPOSICIÓN ALPHA (Alpha Blending)
        Descripción: Combina dos imágenes usando una máscara ponderada
        Fórmula: resultado = foreground * alpha + background * (1 - alpha)
        Ventaja: Mantiene proporciones y escala correctas
    """)


# ============================================================================
# EJEMPLO 8: Procesamiento Frame a Frame Manual
# ============================================================================

def ejemplo_8_frame_a_frame():
    """
    Procesa frames manualmente para mayor control
    """
    print("EJEMPLO 8: Procesamiento Frame a Frame")
    print("-" * 50)
    
    video_path = "video_chroma_key.mp4"
    processor = ChromaKeyProcessor(video_path)
    
    cap = cv2.VideoCapture(video_path)
    
    # Procesar solo 5 frames y mostrar
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar
        mascara = processor.procesar_frame_basico(frame)
        
        # Crear fondo gradiente
        fondo = processor.crear_fondo_gradiente((255, 0, 0), (0, 0, 255))
        
        # Reemplazar fondo
        resultado = processor.reemplazar_fondo(frame, mascara, fondo)
        
        # Mostrar información
        print(f"Frame {i}: Máscara min={mascara.min()}, max={mascara.max()}, "
              f"promedio={mascara.mean():.1f}")
    
    cap.release()
    print("✓ Análisis completado\n")


# ============================================================================
# EJEMPLO 9: Crear Fondos Personalizados
# ============================================================================

def ejemplo_9_fondos_creativos():
    """
    Crea fondos con patrones o diseños personalizados
    """
    print("EJEMPLO 9: Fondos Creativos")
    print("-" * 50)
    
    processor = ChromaKeyProcessor("video_chroma_key.mp4")
    
    # Fondo con patrón de ajedrez
    fondo_ajedrez = np.zeros((processor.height, processor.width, 3), dtype=np.uint8)
    size = 50
    for i in range(0, processor.height, size):
        for j in range(0, processor.width, size):
            if ((i // size) + (j // size)) % 2 == 0:
                fondo_ajedrez[i:i+size, j:j+size] = [255, 255, 255]  # Blanco
            else:
                fondo_ajedrez[i:i+size, j:j+size] = [0, 0, 0]  # Negro
    
    processor.fondo = fondo_ajedrez
    
    processor.procesar_video(
        metodo='basico',
        output_path='resultado_ajedrez.mp4',
        fondo_tipo='imagen'
    )
    print("✓ Video guardado: resultado_ajedrez.mp4\n")


# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EJEMPLOS DE USO - CAMBIO DE FONDO CON CHROMA KEY")
    print("="*60 + "\n")
    
    ejemplos = {
        1: ("Fondo Sólido", ejemplo_1_fondo_solido),
        2: ("Fondo Gradiente", ejemplo_2_fondo_gradiente),
        3: ("Fondo Desenfocado", ejemplo_3_fondo_desenfocado),
        4: ("Fondo con Imagen", ejemplo_4_fondo_imagen),
        5: ("Comparación de Técnicas", ejemplo_5_comparacion),
        6: ("Procesamiento Personalizado", ejemplo_6_personalizado),
        7: ("Explicar Técnicas", explicar_tecnicas),
        8: ("Procesamiento Frame a Frame", ejemplo_8_frame_a_frame),
        9: ("Fondos Creativos", ejemplo_9_fondos_creativos),
    }
    
    print("Ejemplos disponibles:")
    for key, (nombre, _) in ejemplos.items():
        print(f"  {key}. {nombre}")
    
    print("\nNota: Para ejecutar un ejemplo específico, edita esta línea:")
    print("      ejemplo_1_fondo_solido()")
    print("      y cambia '1' por el número del ejemplo que quieras ejecutar\n")
    
    # Descomenta el ejemplo que quieras ejecutar:
    # ejemplo_1_fondo_solido()
    # ejemplo_2_fondo_gradiente()
    # ejemplo_3_fondo_desenfocado()
    # ejemplo_4_fondo_imagen()
    # ejemplo_5_comparacion()
    # ejemplo_6_personalizado()
    # explicar_tecnicas()
    # ejemplo_8_frame_a_frame()
    # ejemplo_9_fondos_creativos()
