"""
Módulo de captura de cámara optimizado para hardware liviano.
"""

import os
import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 1, width: int = 640, height: int = 480):
        self.index = index
        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam en /dev/video{self.index}")

        # Configura resolución de trabajo liviana para el procesador
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Conectado exitosamente a /dev/video{self.index} ({actual_w}x{actual_h})")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Error al capturar frame de la webcam.")
        return frame

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()


def save_background(path: str, frame: np.ndarray) -> None:
    cv2.imwrite(path, frame)


def load_background(path: str) -> np.ndarray:
    return cv2.imread(path) if os.path.exists(path) else None