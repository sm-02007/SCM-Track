"""
Módulo de captura de cámara con detección y selección automática de índice.
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = None):
        """
        Si index es None, busca automáticamente el primer puerto de cámara válido.
        Si se pasa un entero, intenta usar ese índice específico.
        """
        if index is not None:
            self.cap = cv2.VideoCapture(index)
            self.index = index
        else:
            self.cap, self.index = self._find_working_camera()

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir ninguna cámara. "
                f"Sugerencia: verificá permisos de Linux ('sudo usermod -aG video $USER') o la conexión USB."
            )

        print(f"[Camera] Conectado exitosamente a la cámara en /dev/video{self.index}")

    def _find_working_camera(self, max_tested: int = 5) -> tuple[cv2.VideoCapture, int]:
        """Prueba índices de 0 a max_tested y retorna la primera cámara que responda."""
        print("[Camera] Buscando cámaras disponibles...")
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, idx
                cap.release()
        return None, -1

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Error al leer frame de la cámara.")
        return frame

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()


def save_background(path: str, frame: np.ndarray) -> None:
    cv2.imwrite(path, frame)


def load_background(path: str) -> np.ndarray:
    return cv2.imread(path) if cv2.os.path.exists(path) else None