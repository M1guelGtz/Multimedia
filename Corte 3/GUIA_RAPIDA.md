# GUÍA RÁPIDA - Cambio de Fondo de Video con Chroma Key

## 📋 Requisitos Previos

1. **Python 3.7+** instalado
2. **Packages necesarios**:
   ```bash
   pip install opencv-python numpy matplotlib pillow
   ```

## 🎬 Pasos para Usar

### Paso 1: Preparar tu Video
- Ten un video con fondo verde (chroma key)
- Colócalo en la misma carpeta que `QuitarFondoVideo.py`
- O especifica la ruta completa en el código

### Paso 2: Elegir tu Método
El script ofrece **9 opciones de uso**:

| # | Opción | Descripción | Comando |
|---|--------|-------------|---------|
| 1 | Fondo Sólido | Color uniforme (azul, rojo, etc) | `ejemplo_1_fondo_solido()` |
| 2 | Fondo Gradiente | Transición entre dos colores | `ejemplo_2_fondo_gradiente()` |
| 3 | Fondo Desenfocado | Efecto bokeh del frame original | `ejemplo_3_fondo_desenfocado()` |
| 4 | Fondo Imagen | Tu propia imagen como fondo | `ejemplo_4_fondo_imagen()` |
| 5 | Comparación Visual | Ver todas las técnicas en 1 frame | `ejemplo_5_comparacion()` |
| 6 | Personalizado | Control total del procesamiento | `ejemplo_6_personalizado()` |
| 7 | Explicar Técnicas | Detalles de cada método usado | `explicar_tecnicas()` |
| 8 | Frame a Frame | Procesar manualmente cada frame | `ejemplo_8_frame_a_frame()` |
| 9 | Fondos Creativos | Patrones (ajedrez, etc) | `ejemplo_9_fondos_creativos()` |

### Paso 3: Ejecutar

**Opción A - Ejecutar directamente (Quick Start)**:
```python
# En QuitarFondoVideo.py, descomentar al final:
if __name__ == "__main__":
    # Cambiar "video_chroma_key.mp4" por tu archivo
    processor = ChromaKeyProcessor("tu_video.mp4")
    
    # Descomentar UNA de estas líneas:
    processor.procesar_video(metodo='basico', output_path='resultado.mp4', 
                            fondo_tipo='color', fondo_valor=(255,0,0))
```

**Opción B - Usar ejemplos**:
```bash
python ejemplos_uso.py
```

## 🛠️ Técnicas Utilizadas

Todas estas técnicas están implementadas y combinadas en el código:

### 1. **Segmentación por Color (HSV)**
   - Detecta píxeles verdes en espacio HSV
   - Más robusto a cambios de iluminación
   - `segmentar_verde_hsv()`

### 2. **Segmentación por Canales RGB**
   - Detecta verde usando: `G > R * 1.2` y `G > B * 1.2`
   - Métrica similar a la que usaste en flower_segmentation.py
   - `segmentar_verde_rgb()`

### 3. **Operaciones Morfológicas**
   - **Erosión**: Reduce objetos blancos (elimina ruido)
   - **Dilatación**: Expande objetos blancos (rellena agujeros)
   - **Apertura**: Erosión + Dilatación (limpieza)
   - **Cierre**: Dilatación + Erosión (llenado)
   - Métodos: `erosion()`, `dilatacion()`, `apertura()`, `cierre()`

### 4. **Suavizado Gaussiano**
   - Suaviza bordes para transiciones naturales
   - Evita artefactos mediante `GaussianBlur()`
   - `suavizar_mascara()`

### 5. **Detección de Bordes**
   - **Sobel**: Detecta bordes mediante derivadas
   - **Laplaciano**: Detección más sensible
   - Método: `aplicar_filtro_sobel()`, `aplicar_filtro_laplaciano()`

### 6. **Alpha Blending (Composición)**
   - Combina foreground y background usando:
   - `resultado = fg * máscara + bg * (1 - máscara)`
   - Mantiene proporciones correctas

## 📊 Comparación de Métodos

### Método Básico (`metodo='basico'`)
- ✓ Más rápido
- ✓ Bueno para fondos uniformes
- ✗ Puede dejar artefactos en bordes
```python
processor.procesar_video(metodo='basico', ...)
```

### Método Avanzado (`metodo='avanzado'`)
- ✓ Mejor calidad de bordes
- ✓ Más limpio en detalles
- ✗ Más lento (más operaciones)
```python
processor.procesar_video(metodo='avanzado', ...)
```

## 🎨 Tipos de Fondo

### 1. Color Sólido
```python
fondo_tipo='color',
fondo_valor=(255, 0, 0)  # BGR: Azul
```

### 2. Gradiente
```python
fondo_tipo='gradiente'
# Automáticamente crea gradiente rojo-amarillo
```

### 3. Desenfoque (Bokeh)
```python
fondo_tipo='desenfoque'
# Desenfoca el frame original como fondo
```

### 4. Imagen Personal
```python
processor = ChromaKeyProcessor("video.mp4", fondo_path="mi_fondo.jpg")
processor.procesar_video(fondo_tipo='imagen', ...)
```

## 🎯 Ejemplo Rápido Completo

```python
from QuitarFondoVideo import ChromaKeyProcessor

# Crear procesador
processor = ChromaKeyProcessor("video.mp4")

# Procesar con fondo azul
processor.procesar_video(
    metodo='avanzado',           # Mejor calidad
    output_path='resultado.mp4', # Archivo de salida
    fondo_tipo='color',          # Tipo de fondo
    fondo_valor=(255, 0, 0)      # BGR: Azul
)

# Ver comparación de técnicas
processor.mostrar_comparacion_frame(frame=50)
```

## 📈 Parámetros Ajustables

Dentro de `QuitarFondoVideo.py` puedes ajustar:

```python
# Rangos de detección HSV (línea ~43)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Tamaño de kernel para operaciones morfológicas (línea ~58, etc)
self.apertura(mascara, kernel_size=7)  # Aumentar = más suave

# Factor de suavizado (línea ~113)
self.suavizar_mascara(mascara, blur_kernel=15)  # 15, 21, 31...

# Umbral de diferencia RGB (línea ~34)
umbral_diff = 60  # Aumentar = más estricto
```

## 🔍 Depuración

### Ver información del video
```python
processor = ChromaKeyProcessor("video.mp4")
print(f"FPS: {processor.fps}")
print(f"Resolución: {processor.width}x{processor.height}")
print(f"Frames totales: {processor.total_frames}")
```

### Verificar máscara en un frame específico
```python
processor.mostrar_comparacion_frame(num_frame=100)
# Muestra 9 vistas diferentes del procesamiento
```

### Estadísticas de la máscara
```python
mascara = processor.procesar_frame_basico(frame)
print(f"Mín: {mascara.min()}")
print(f"Máx: {mascara.max()}")
print(f"Promedio: {mascara.mean()}")
```

## ⚙️ Configuración Recomendada

| Situación | Método | Fondo | Kernel |
|-----------|--------|-------|--------|
| Video profesional | avanzado | imagen | 7 |
| Fondo simple | basico | color | 5 |
| Movimiento rápido | basico | color/gradiente | 5 |
| Detalles finos | avanzado | desenfoque/imagen | 9 |
| Presupuesto bajo (rápido) | basico | color | 3 |

## 📁 Archivos Generados

Después de ejecutar, encontrarás:
- `video_fondo_azul.mp4` - Con fondo azul
- `video_fondo_gradiente.mp4` - Con gradiente
- `video_fondo_desenfoque.mp4` - Con bokeh
- Y más según los ejemplos ejecutados

## 🐛 Problemas Comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| "No se encontró el video" | Ruta incorrecta | Especifica ruta completa o pon archivo en misma carpeta |
| Bordes dentados | Máscara pobre | Aumentar `blur_kernel` o cambiar rangos HSV |
| Muy lento | Método avanzado | Usar `metodo='basico'` o reducir resolución |
| Agujeros en la persona | Segmentación mala | Ajustar `lower_green` y `upper_green` |
| Fondo visible en bordes | Máscara incompleta | Usar operación `cierre()` con kernel mayor |

## 📚 Archivos Principales

- **QuitarFondoVideo.py** - Clase principal `ChromaKeyProcessor`
- **ejemplos_uso.py** - 9 ejemplos listos para usar
- **GUIA_RAPIDA.md** - Este archivo

## 🚀 Próximas Mejoras Posibles

1. Detección automática de rango HSV
2. Predicción de movimiento para frames siguientes
3. Machine Learning para segmentación (U-Net, etc)
4. Interpolación de bordes
5. Video en tiempo real (webcam)

---

**¡Listo para empezar! 🎬**

Descomentar un ejemplo en `ejemplos_uso.py` y ejecutar:
```bash
python ejemplos_uso.py
```

O usar directamente en tu código:
```python
from QuitarFondoVideo import ChromaKeyProcessor
processor = ChromaKeyProcessor("tu_video.mp4")
processor.procesar_video(...)
```
