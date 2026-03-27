import importlib
import numpy as np
from pathlib import Path
from typing import Any, Callable, cast

cv2_api = cast(Any, importlib.import_module("cv2"))

_cv2_imshow = cast(Callable[[str, np.ndarray], None], getattr(cv2_api, "imshow"))
_cv2_wait_key = cast(Callable[[int], int], getattr(cv2_api, "waitKey"))
_cv2_destroy_all_windows = cast(Callable[[], None], getattr(cv2_api, "destroyAllWindows"))

class BallTracker:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.lower_color = np.array([35, 80, 80])
        self.upper_color = np.array([85, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2_api.cvtColor(frame, cv2_api.COLOR_BGR2HSV)
        mask = cv2_api.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2_api.morphologyEx(mask, cv2_api.MORPH_OPEN, self.kernel)
        mask = cv2_api.dilate(mask, self.kernel, iterations=1)
        return cv2_api.bitwise_and(frame, frame, mask=mask)

    def _setup_video_writer(self, cap: Any) -> Any:
        fps = cap.get(cv2_api.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2_api.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2_api.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2_api.VideoWriter_fourcc(*'mp4v')
        out = cv2_api.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        return out

    def process_video(self, show_preview: bool = True) -> bool:
        if not self.input_path.exists():
            print(f"No se encontró el video de entrada: {self.input_path}")
            return False

        cap = cv2_api.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {self.input_path}")
            return False

        out = self._setup_video_writer(cap)
        if not out.isOpened():
            print(f"No se pudo crear el video de salida: {self.output_path}")
            cap.release()
            return False

        frame_count = 0
        total_frames = int(cap.get(cv2_api.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self._process_frame(frame)
            out.write(result)

            if show_preview:
                _cv2_imshow("Seguimiento de pelota", result)
                key = _cv2_wait_key(1) & 0xFF
                window_open = cv2_api.getWindowProperty("Seguimiento de pelota", cv2_api.WND_PROP_VISIBLE) >= 1
                if key in (ord("q"), 27) or not window_open:
                    break

            frame_count += 1
            if total_frames > 0 and frame_count >= total_frames:
                break

        cap.release()
        out.release()
        if show_preview:
            _cv2_destroy_all_windows()

        print(f"Frames procesados: {frame_count}")
        print(f"Video de salida guardado en: {self.output_path}")
        return True

    def display_output(self) -> None:
        if not self.output_path.exists():
            print(f"No existe el video de salida: {self.output_path}")
            return

        cap = cv2_api.VideoCapture(str(self.output_path))
        if not cap.isOpened():
            print(f"No se pudo abrir el video de salida: {self.output_path}")
            return

        frame_count = 0
        total_frames = int(cap.get(cv2_api.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _cv2_imshow("Resultado final", frame)
            key = _cv2_wait_key(25) & 0xFF
            window_open = cv2_api.getWindowProperty("Resultado final", cv2_api.WND_PROP_VISIBLE) >= 1
            if key in (ord("q"), 27) or not window_open:
                break

            frame_count += 1
            if total_frames > 0 and frame_count >= total_frames:
                break

        cap.release()
        _cv2_destroy_all_windows()

if __name__ == "__main__":
    tracker = BallTracker('./Corte 3/ball_tracking_example.mp4', './Corte 3/ball_tracking_output.mp4')

    if tracker.process_video(show_preview=False):
        tracker.display_output()