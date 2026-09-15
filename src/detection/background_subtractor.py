"""
Background Subtractor insensible a sombras y variaciones de luz ambiente (Espacio HSV).
"""

import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, threshold: int = 30, blur_ksize: int = 7, sat_weight: float = 0.85, val_weight: float = 0.15):
        self.threshold = threshold
        self.blur_ksize = blur_ksize
        self.sat_weight = sat_weight
        self.val_weight = val_weight
        self.background_hsv = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Blur gaussiano más amplio para dispersar reflejos brillantes
        blurred = cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    def set_background(self, frame_bgr: np.ndarray) -> None:
        self.background_hsv = self._preprocess(frame_bgr)

    def has_background(self) -> bool:
        return self.background_hsv is not None

    def compute_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.background_hsv is None:
            raise RuntimeError("Background no capturado. Llamá a set_background() primero.")

        current_hsv = self._preprocess(frame_bgr)

        if self.background_hsv.shape != current_hsv.shape:
            self.set_background(frame_bgr)
            current_hsv = self._preprocess(frame_bgr)

        # Diferencia de Saturación (S) y Valor/Brillo (V) descartando cambios puros de tono/sombra
        diff_hsv = cv2.absdiff(self.background_hsv, current_hsv)
        diff_sat = diff_hsv[:, :, 1]
        diff_val = diff_hsv[:, :, 2]

        # Saturación pesa más: es más robusta a sombras/brillo que Valor.
        # Ajustables en vivo vía sat_weight/val_weight (ej. trackbar).
        combined_diff = cv2.addWeighted(diff_sat, self.sat_weight, diff_val, self.val_weight, 0)

        _, mask = cv2.threshold(combined_diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask