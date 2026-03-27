"""
TÉCNICAS AVANZADAS - Procesamiento Avanzado de Chroma Key

Este archivo contiene técnicas más avanzadas para mejorar la calidad
del cambio de fondo del video.
"""

import cv2
import numpy as np
from QuitarFondoVideo import ChromaKeyProcessor
import matplotlib.pyplot as plt


# ============================================================================
# TÉCNICA 1: Detección Automática del Rango de Color
# ============================================================================

class ChromaKeyAutomatico(ChromaKeyProcessor):
    """
    Extiende ChromaKeyProcessor con detección automática de color chroma
    """
    
    def detectar_rango_chroma_automatico(self, num_frames=10):
        """
        Analiza los primeros N frames para encontrar automáticamente
        el rango de color más común (probablemente el fondo verde)
        """
        print(f"Analizando {num_frames} frames para detectar rango chroma...")
        
        valores_hsv = []
        
        for frame_num in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Tomar píxeles de las esquinas (probablemente sean fondo)
            esquina = hsv[0:100, 0:100]
            valores_hsv.append(esquina.reshape(-1, 3))
        
        todos_hsv = np.vstack(valores_hsv)
        
        # Calcular estadísticas
        h_mean, s_mean, v_mean = todos_hsv.mean(axis=0)
        h_std, s_std, v_std = todos_hsv.std(axis=0)
        
        print(f"\nRango detectado:")
        print(f"  H: {h_mean:.0f} ± {h_std:.0f}")
        print(f"  S: {s_mean:.0f} ± {s_std:.0f}")
        print(f"  V: {v_mean:.0f} ± {v_std:.0f}")
        
        # Usar 2 desviaciones estándar
        lower = np.array([
            max(0, h_mean - 2*h_std),
            max(0, s_mean - 2*s_std),
            max(0, v_mean - 2*v_std)
        ])
        upper = np.array([
            min(180, h_mean + 2*h_std),
            min(255, s_mean + 2*s_std),
            min(255, v_mean + 2*v_std)
        ])
        
        return lower, upper


# ============================================================================
# TÉCNICA 2: Suavizado de Bordes con Anti-aliasing
# ============================================================================

def aplicar_antialiasing(image, mask):
    """
    Aplica anti-aliasing para suavizar los bordes de la composición
    """
    # Convertir máscara a espacio de 0-1
    alpha = mask.astype(float) / 255.0
    
    # Aplicar filtro Gaussiano iterativo para suavizado
    for _ in range(3):
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # Normalizar nuevamente
    alpha = (alpha * 255).astype(np.uint8)
    return alpha


# ============================================================================
# TÉCNICA 3: Corrección de Iluminación en Bordes
# ============================================================================

def corregir_iluminacion_bordes(frame, mascara, extension=20):
    """
    Corrige la iluminación en los bordes del objeto
    para una transición más natural
    """
    # Crear máscara expandida
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extension, extension))
    mascara_expandida = cv2.dilate(mascara, kernel, iterations=1)
    
    # Crear zona de transición
    zona_transicion = mascara_expandida - mascara
    
    # Aplicar desenfoque solo en la zona de transición
    frame_desenfocado = cv2.GaussianBlur(frame, (15, 15), 0)
    
    # Mezclar
    resultado = frame.copy()
    resultado[zona_transicion > 0] = (
        frame[zona_transicion > 0] * 0.5 + 
        frame_desenfocado[zona_transicion > 0] * 0.5
    )
    
    return resultado


# ============================================================================
# TÉCNICA 4: Composición Con Bordes Suaves (Edge Preservation)
# ============================================================================

def composicion_bordes_suaves(frame, mascara, fondo, sigma=1.0):
    """
    Realiza composición preservando los bordes naturales
    """
    # Normalizar máscara
    alpha = mascara.astype(float) / 255.0
    
    # Aplicar filtro Gaussiano se ñalado en alpha
    # Esto crea transiciones suaves en los bordes
    alpha_smooth = cv2.GaussianBlur(alpha, (21, 21), sigma)
    
    # Expandir a 3 canales
    alpha_smooth = np.dstack([alpha_smooth] * 3)
    
    # Composición
    resultado = (
        frame.astype(float) * alpha_smooth + 
        fondo.astype(float) * (1 - alpha_smooth)
    ).astype(np.uint8)
    
    return resultado


# ============================================================================
# TÉCNICA 5: Aplicar Sombra para Realismo
# ============================================================================

def agregar_sombra_realista(resultado, mascara, fondo, intensidad=0.3):
    """
    Agrega una sombra suave del objeto en el fondo
    para mayor realismo
    """
    # Crear sombra desenfocada
    mascara_sombra = cv2.GaussianBlur(mascara.astype(float), (51, 51), 20)
    mascara_sombra = mascara_sombra / mascara_sombra.max()
    
    # Oscurecer el fondo donde está la sombra
    fondo_sombra = fondo.copy().astype(float)
    
    # Expandir máscara a 3 canales
    mascara_sombra_3ch = np.dstack([mascara_sombra] * 3)
    
    # Aplicar sombra
    fondo_sombra = fondo_sombra * (1 - mascara_sombra_3ch * intensidad)
    
    # Combinar
    resultado_final = resultado.astype(float) * 0.7 + fondo_sombra.astype(float) * 0.3
    
    return resultado_final.astype(np.uint8)


# ============================================================================
# TÉCNICA 6: Multiplicar Iluminación del Objeto
# ============================================================================

def multiplicar_iluminacion(frame, mascara):
    """
    Ajusta la iluminación del objeto basada en el valor de la máscara
    """
    # Crear factor de iluminación suave
    alpha = mascara.astype(float) / 255.0
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    alpha = np.dstack([alpha] * 3)
    
    # Multiplicar iluminación (brightening where mask is strong)
    factor_luz = 1.0 + alpha * 0.2  # +20% brightness
    
    frame_iluminado = (frame.astype(float) * factor_luz).astype(np.uint8)
    frame_iluminado = np.clip(frame_iluminado, 0, 255)
    
    return frame_iluminado


# ============================================================================
# TÉCNICA 7: Filtro Bilateral (Borde-Preservador)
# ============================================================================

def aplicar_bilateral_filter(frame, mascara):
    """
    Aplica filtro bilateral para suavizar manteniendo bordes
    """
    # Aplicar en la región donde está el objeto
    resultado = cv2.bilateralFilter(frame, 9, 75, 75)
    return resultado


# ============================================================================
# TÉCNICA 8: Segmentación Inteligente Multi-Canal
# ============================================================================

def segmentacion_multichannel(frame):
    """
    Combina múltiples criterios para segmentación más robusta
    """
    # Canal HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_hsv = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    
    # Canal LAB (alternativa)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mascara_lab = cv2.inRange(lab, np.array([0, 0, 0]), np.array([255, 120, 140]))
    
    # Combinar (AND)
    mascara_combinada = cv2.bitwise_and(mascara_hsv, mascara_lab)
    
    return mascara_combinada


# ============================================================================
# TÉCNICA 9: Estimación del Fondo Adaptativo
# ============================================================================

class FondoAdaptativo:
    """
    Crea un fondo que se adapta al contenido del video
    """
    
    @staticmethod
    def fondo_blur_adaptativo(frame, intensidad=0.7):
        """
        Crea fondo borroso que se adapta a los colores del frame
        """
        # Versión muy desenfocada
        fondo_base = cv2.blur(frame, (101, 101))
        
        # Versión muy coloreada
        fondo_hsv = cv2.cvtColor(fondo_base, cv2.COLOR_BGR2HSV)
        fondo_hsv[:, :, 1] = cv2.multiply(fondo_hsv[:, :, 1], 1.5)  # Saturación
        fondo_final = cv2.cvtColor(fondo_hsv, cv2.COLOR_HSV2BGR)
        
        return fondo_final
    
    @staticmethod
    def fondo_dominante(frame):
        """
        Extrae el color dominante del frame como fondo
        """
        # Redimensionar para análisis rápido
        pequeno = cv2.resize(frame, (150, 150))
        
        # Contar píxeles
        pixeles = pequeno.reshape((-1, 3))
        pixeles = np.float32(pixeles)
        
        # K-means para encontrar color dominante
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixeles, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        color_dominante = np.uint8(centers[0])
        
        # Crear fondo sólido
        fondo = np.zeros_like(frame)
        fondo[:] = color_dominante
        
        return fondo


# ============================================================================
# TÉCNICA 10: Ajuste de Contraste Inteligente
# ============================================================================

def ajustar_contraste_inteligente(resultado, target_brightness=127):
    """
    Ajusta el contraste de la composición final
    """
    # Convertir a escala de grises para calcular brillo
    gris = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
    brillo_actual = gris.mean()
    
    # Calcular factor de ajuste
    factor = target_brightness / (brillo_actual + 1)
    
    # Aplicar CLAHE (Contraste Adaptativo Limitado)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_mejorado = clahe.apply(gris)
    
    # Aplicar al resultado RGB
    resultado_mejorado = result.copy()
    for i in range(3):
        resultado_mejorado[:, :, i] = cv2.multiply(resultado[:, :, i], factor)
    
    return np.clip(resultado_mejorado, 0, 255).astype(np.uint8)


# ============================================================================
# EJEMPLO COMPLETO: Usar Todas las Técnicas Avanzadas
# ============================================================================

def procesar_video_con_tecnicas_avanzadas(video_path, output_path):
    """
    Ejemplo completo usando todas las técnicas avanzadas
    """
    print("Procesando video con técnicas avanzadas...")
    
    processor = ChromaKeyProcessor(video_path)
    cap = cv2.VideoCapture(video_path)
    
    # Crear writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = output_path.replace('.mp4', '.avi')
    out = cv2.VideoWriter(output_path, fourcc, processor.fps, 
                         (processor.width, processor.height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Segmentación multichannel
        mascara = segmentacion_multichannel(frame)
        mascara = processor.apertura(mascara, kernel_size=7)
        mascara = processor.cierre(mascara, kernel_size=7)
        
        # 2. Suavizar con antialiasing
        mascara = aplicar_antialiasing(frame, mascara)
        
        # 3. Crear fondo adaptativo
        fondo = FondoAdaptativo.fondo_blur_adaptativo(frame)
        
        # 4. Composición con bordes suaves
        resultado = composicion_bordes_suaves(frame, mascara, fondo, sigma=1.5)
        
        # 5. Agregar sombra
        resultado = agregar_sombra_realista(resultado, mascara, fondo, intensidad=0.2)
        
        # Escribir frame
        out.write(resultado)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{processor.total_frames}")
    
    cap.release()
    out.release()
    print(f"✓ Video guardado: {output_path}")


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TÉCNICAS AVANZADAS - CHROMA KEY")
    print("="*60 + "\n")
    
    print("""
    Técnicas implementadas en este módulo:
    
    1. Detección Automática de Color Chroma
    2. Anti-aliasing en Bordes
    3. Corrección de Iluminación en Bordes
    4. Composición con Bordes Suaves
    5. Agregación de Sombras Realistas
    6. Multiplicación de Iluminación
    7. Filtro Bilateral (Edge-Preserving)
    8. Segmentación Multi-canal
    9. Fondo Adaptativo
    10. Ajuste de Contraste Inteligente
    
    Para usar estas técnicas avanzadas, descomentar:
    - procesar_video_con_tecnicas_avanzadas("video.mp4", "salida.mp4")
    """)
    
    # Descomentar para usar:
    # procesar_video_con_tecnicas_avanzadas("video.mp4", "video_avanzado.mp4")
