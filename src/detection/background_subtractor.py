"""
Background Subtractor adaptable a cualquier superficie.
Convierte a HSV para analizar luminancia y saturación por separado,
tolerando cambios leves de iluminación según el fondo.
"""

import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, threshold: int = 25, blur_ksize: int = 7):
        self.threshold = threshold
        self.blur_ksize = blur_ksize
        self.background_bgr = None
        self.background_hsv = None

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blurred_bgr = cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)
        blurred_hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
        return blurred_bgr, blurred_hsv

    def set_background(self, frame_bgr: np.ndarray) -> None:
        self.background_bgr, self.background_hsv = self._preprocess(frame_bgr)

    def has_background(self) -> bool:
        return self.background_bgr is not None

    def compute_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.background_bgr is None:
            raise RuntimeError("Background no capturado. Llamá a set_background() primero.")

        current_bgr, current_hsv = self._preprocess(frame_bgr)

        # 1. Diferencia en BGR (canal por canal)
        diff_bgr = cv2.absdiff(self.background_bgr, current_bgr)
        diff_bgr_max = np.max(diff_bgr, axis=2)

        # 2. Diferencia en HSV (Matiz/Saturación) para fondos de color uniforme como hierro gris
        diff_hsv = cv2.absdiff(self.background_hsv, current_hsv)
        diff_sat = diff_hsv[:, :, 1]  # Saturación
        diff_val = diff_hsv[:, :, 2]  # Valor / Brillo

        # Combinación de diferencias para adaptarse a superficies claras, oscuras o metálicas
        combined_diff = cv2.addWeighted(diff_bgr_max, 0.6, cv2.addWeighted(diff_sat, 0.2, diff_val, 0.2, 0), 0.4, 0)

        _, mask = cv2.threshold(combined_diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask