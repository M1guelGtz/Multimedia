"""
Detector de puntos de dados en video.
Utiliza binarizado por color (RGB) para detectar dados rojos
y contar sus puntos blancos, mostrando la suma total.
Sin usar OpenCV - solo NumPy, SciPy, Pillow e imageio.
"""

import numpy as np
from scipy import ndimage
from collections import Counter
import imageio.v3 as iio
import imageio
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
import time
import os


def detectar_dados(frame):
    """
    Detecta dados rojos usando binarizado en RGB.
    Retorna lista de bounding boxes (x1, y1, x2, y2).
    """
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)

    # Binarizado: rojo = R alto, G y B bajos
    mask_rojo = (r > 120) & (g < 100) & (b < 100)

    # Cerrar huecos (los puntos blancos crean huecos en la mascara roja)
    struct = ndimage.generate_binary_structure(2, 2)
    mask_cerrada = ndimage.binary_closing(mask_rojo, structure=struct, iterations=10)
    mask_cerrada = ndimage.binary_fill_holes(mask_cerrada)

    # Etiquetar componentes conexos
    etiquetas, n_componentes = ndimage.label(mask_cerrada)

    dados = []
    for i in range(1, n_componentes + 1):
        componente = etiquetas == i
        area = np.count_nonzero(componente)
        if area < 1500 or area > 20000:
            continue
        coords = np.argwhere(componente)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        w, h = x2 - x1, y2 - y1
        aspect = w / float(h) if h > 0 else 0
        if 0.5 < aspect < 2.0:
            dados.append((x1, y1, x2, y2))

    return dados


def contar_puntos_blancos(frame, x1, y1, x2, y2):
    """
    Cuenta puntos blancos dentro de la region de un dado
    usando binarizado por umbral de color blanco en RGB.
    """
    roi = frame[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return 0

    roi_r = roi[:, :, 0].astype(int)
    roi_g = roi[:, :, 1].astype(int)
    roi_b = roi[:, :, 2].astype(int)

    # Binarizado: blanco = R, G y B todos altos
    mask_blanco = (roi_r > 180) & (roi_g > 180) & (roi_b > 180)

    # Limpiar ruido
    mask_blanco = ndimage.binary_opening(mask_blanco, iterations=1)

    # Etiquetar y contar componentes
    etiq_puntos, n_puntos = ndimage.label(mask_blanco)

    puntos_validos = 0
    for j in range(1, n_puntos + 1):
        a = np.count_nonzero(etiq_puntos == j)
        if a > 5:
            puntos_validos += 1

    return min(puntos_validos, 6)


def anotar_frame(frame, dados, resultados, suma_total, frame_num, total_frames):
    """Dibuja anotaciones sobre el frame usando Pillow."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font_grande = ImageFont.truetype("arial.ttf", 40)
        font_medio = ImageFont.truetype("arial.ttf", 32)
        font_chico = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font_grande = ImageFont.load_default()
        font_medio = font_grande
        font_chico = font_grande

    # Rectangulo y valor por cada dado
    for idx, (x1, y1, x2, y2) in enumerate(dados):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        if idx < len(resultados):
            draw.text((x1, y1 - 45), str(resultados[idx]),
                      fill=(0, 255, 0), font=font_grande)

    # Panel superior negro
    ancho = frame.shape[1]
    draw.rectangle([0, 0, ancho, 100], fill=(0, 0, 0))
    draw.text((10, 10), f"Dados detectados: {len(dados)}",
              fill=(0, 255, 255), font=font_medio)

    if resultados:
        detalle = " + ".join(map(str, resultados)) + f" = {suma_total}"
        draw.text((10, 55), f"Suma: {detalle}",
                  fill=(255, 255, 255), font=font_medio)

    # Info de frame
    draw.text((ancho - 300, frame.shape[0] - 35),
              f"Frame: {frame_num}/{total_frames}",
              fill=(200, 200, 200), font=font_chico)

    return np.array(img)


def procesar_video(ruta_video):
    """Procesa el video, detecta dados rojos y cuenta sus puntos blancos."""
    print("Cargando video...")
    frames = iio.imread(ruta_video, plugin="pyav")
    total_frames = frames.shape[0]
    alto, ancho = frames.shape[1], frames.shape[2]
    fps = 30

    print(f"Video: {ancho}x{alto}, {total_frames} frames")
    print("Procesando frames...")
    print("-" * 50)

    historial_sumas = []
    frames_anotados = []

    for i in range(total_frames):
        frame = frames[i]
        dados = detectar_dados(frame)

        suma_total = 0
        resultados = []

        for (x1, y1, x2, y2) in dados:
            puntos = contar_puntos_blancos(frame, x1, y1, x2, y2)
            suma_total += puntos
            resultados.append(puntos)

        if len(dados) > 0 and suma_total > 0:
            historial_sumas.append(suma_total)

        frame_anotado = anotar_frame(frame, dados, resultados,
                                     suma_total, i + 1, total_frames)
        frames_anotados.append(frame_anotado)

    # Guardar video
    ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "tirada_1_resultado.mp4")
    print(f"\nGuardando video en: {ruta_salida}")
    writer = imageio.get_writer(ruta_salida, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=1)
    for fa in frames_anotados:
        writer.append_data(fa)
    writer.close()
    print("Video guardado.")

    # Moda de las sumas
    if historial_sumas:
        conteo = Counter(historial_sumas)
        suma_final = conteo.most_common(1)[0][0]
    else:
        suma_final = 0

    print(f"\nSuma total de los dados: {suma_final}")

    # Mostrar video en ventana tkinter
    print("\nReproduciendo video (cierra la ventana para salir)...")
    mostrar_video(frames_anotados, fps, ancho, alto)


def mostrar_video(frames_anotados, fps, ancho, alto):
    """Reproduce el video anotado en una ventana tkinter."""
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Detector de Dados")

    # Escalar para que quepa en pantalla
    escala = min(1.0, 800 / alto)
    new_w = int(ancho * escala)
    new_h = int(alto * escala)

    canvas = tk.Canvas(root, width=new_w, height=new_h)
    canvas.pack()

    delay = int(1000 / fps)
    idx = [0]
    photo_ref = [None]

    def actualizar():
        if idx[0] >= len(frames_anotados):
            return
        img = Image.fromarray(frames_anotados[idx[0]])
        img = img.resize((new_w, new_h), Image.LANCZOS)
        photo_ref[0] = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo_ref[0])
        idx[0] += 1
        if idx[0] < len(frames_anotados):
            root.after(delay, actualizar)

    actualizar()
    root.mainloop()


if __name__ == "__main__":
    ruta = r"C:\Users\Migue\Downloads\tirada_1.mp4"
    procesar_video(ruta)
