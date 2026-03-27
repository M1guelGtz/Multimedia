import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ChromaKeyProcessor:
    def __init__(self, video_path, fondo_path=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fondo = None
        if fondo_path:
            img = cv2.imread(fondo_path)
            if img is not None:
                self.fondo = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def _reset_cap(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)

    # Tecnica 1: Segmentacion HSV
    def mask_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Tecnica 2: Segmentacion RGB
    def mask_rgb(self, frame):
        b, g, r = cv2.split(frame)
        return np.where((g > r * 1.2) & (g > b * 1.2) & (g > 100), 255, 0).astype(np.uint8)

    # Tecnicas 3-6: Morfologia (erosion, dilatacion, apertura y cierre)
    def morph(self, img, op, k=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(img, op, kernel)

    # Tecnica 7: Suavizado Gaussiano
    def smooth(self, mask, k=15):
        return cv2.GaussianBlur(mask, (k, k), 0)

    # Tecnica 8: Sobel
    def sobel(self, gray):
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.clip(np.sqrt(sx * sx + sy * sy), 0, 255).astype(np.uint8)

    # Tecnica 9: Laplaciano
    def laplaciano(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F)

    def mask_basica(self, frame):
        m = cv2.bitwise_not(self.mask_hsv(frame))
        m = self.morph(m, cv2.MORPH_OPEN, 7)
        m = self.morph(m, cv2.MORPH_CLOSE, 7)
        return self.smooth(m, 15)

    def mask_avanzada(self, frame):
        mh = self.mask_hsv(frame)
        mr = self.mask_rgb(frame)
        m = cv2.bitwise_not(cv2.bitwise_or(mh, mr))
        m = self.morph(m, cv2.MORPH_DILATE, 5)
        m = self.morph(m, cv2.MORPH_ERODE, 5)
        m = self.morph(m, cv2.MORPH_CLOSE, 7)
        return self.smooth(m, 21)

    def componer(self, frame, mask, fondo=None):
        bg = fondo if fondo is not None else self.fondo
        if bg is None:
            bg = np.full_like(frame, (255, 0, 0), dtype=np.uint8)
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        a = (mask.astype(np.float32) / 255.0)[..., None]
        return (frame * a + bg * (1.0 - a)).astype(np.uint8)

    def fondo_gradiente(self, c1=(0, 0, 255), c2=(255, 0, 0)):
        t = np.linspace(0, 1, self.height, dtype=np.float32)[:, None, None]
        top = np.array(c1, dtype=np.float32)[None, None, :]
        bottom = np.array(c2, dtype=np.float32)[None, None, :]
        grad = (top * (1 - t) + bottom * t).astype(np.uint8)
        return np.repeat(grad, self.width, axis=1)

    def cargar_fondos(self, carpeta, n=7):
        ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        rutas = [
            os.path.join(carpeta, f)
            for f in sorted(os.listdir(carpeta))
            if os.path.isfile(os.path.join(carpeta, f)) and f.lower().endswith(ext)
        ]
        imgs = []
        for r in rutas:
            im = cv2.imread(r)
            if im is not None:
                imgs.append(cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA))
        if not imgs:
            raise FileNotFoundError(f"No hay imagenes validas en: {carpeta}")
        return [imgs[i % len(imgs)] for i in range(n)]

    def procesar_video(self, metodo="basico", output_path="video_procesado.avi", fondo_tipo="carpeta",
                       fondo_valor=(100, 200, 255), carpeta_fondos=None, total_fondos_escena=7):
        self._reset_cap()
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if not output_path.lower().endswith(".avi"):
            output_path = output_path.rsplit(".", 1)[0] + ".avi"
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        fondos = None
        frames_por_escena = 1
        if fondo_tipo == "carpeta":
            if not carpeta_fondos:
                raise ValueError("Debes indicar carpeta_fondos cuando fondo_tipo='carpeta'")
            fondos = self.cargar_fondos(carpeta_fondos, total_fondos_escena)
            frames_por_escena = max(1, self.total_frames // len(fondos))
            print(f"Fondos usados: {len(fondos)} | Frames por escena: {frames_por_escena}")

        t0, i = time.time(), 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            mask = self.mask_basica(frame) if metodo == "basico" else self.mask_avanzada(frame)

            if fondo_tipo == "carpeta":
                bg = fondos[min(i // frames_por_escena, len(fondos) - 1)]
            elif fondo_tipo == "imagen":
                bg = self.fondo
            elif fondo_tipo == "gradiente":
                bg = self.fondo_gradiente((0, 0, 255), (255, 255, 0))
            elif fondo_tipo == "desenfoque":
                bg = cv2.blur(frame, (51, 51))
            else:
                bg = np.full_like(frame, fondo_valor, dtype=np.uint8)

            out.write(self.componer(frame, mask, bg))
            i += 1
            if i % 30 == 0:
                dt = time.time() - t0
                print(f"Frame {i}/{self.total_frames} | FPS {i / max(dt, 1e-6):.1f}")

        out.release()
        self.cap.release()
        print(f"\nVideo listo en {time.time() - t0:.2f}s")
        print(f"Guardado en: {output_path}")

    def mostrar_comparacion_frame(self, num_frame=50):
        self._reset_cap()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
        ok, frame = self.cap.read()
        if not ok:
            print(f"No se pudo leer el frame {num_frame}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m_hsv, m_rgb = self.mask_hsv(frame), self.mask_rgb(frame)
        m_b, m_a = self.mask_basica(frame), self.mask_avanzada(frame)
        sbl, lap = self.sobel(gray), self.laplaciano(gray)
        r_b, r_a = self.componer(frame, m_b), self.componer(frame, m_a)

        imgs = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), m_hsv, m_rgb,
            m_b, m_a, sbl,
            cv2.cvtColor(r_b, cv2.COLOR_BGR2RGB), cv2.cvtColor(r_a, cv2.COLOR_BGR2RGB), lap,
        ]
        tit = [
            "Original", "Mascara HSV", "Mascara RGB",
            "Mascara Basica", "Mascara Avanzada", "Sobel",
            "Resultado Basico", "Resultado Avanzado", "Laplaciano",
        ]
        cmaps = [None, "gray", "gray", "gray", "gray", "gray", None, None, "gray"]

        plt.figure(figsize=(16, 10))
        for k, (im, t, cm) in enumerate(zip(imgs, tit, cmaps), 1):
            plt.subplot(3, 3, k)
            plt.imshow(im, cmap=cm)
            plt.title(t)
            plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    VIDEO_PATH = "./Corte 3/Man_walking_with_green_screen_background.mp4"
    CARPETA_FONDOS = "./Corte 3/fondos"

    try:
        p = ChromaKeyProcessor(VIDEO_PATH)
        p.procesar_video(
            metodo="avanzado",
            output_path="video_fondos_7_escenas.avi",
            fondo_tipo="carpeta",
            carpeta_fondos=CARPETA_FONDOS,
            total_fondos_escena=7,
        )
        p.mostrar_comparacion_frame(50)
    except FileNotFoundError as e:
        print(f"Error: {e}")
