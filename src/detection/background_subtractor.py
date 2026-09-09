"""
Background Subtractor ultra liviano.
Procesa las diferencias directamente en escala BGR de baja resolución.
"""

import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, threshold: int = 25, blur_ksize: int = 5):
        self.threshold = threshold
        self.blur_ksize = blur_ksize
        self.background = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Blur liviano
        return cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)

    def set_background(self, frame_bgr: np.ndarray) -> None:
        self.background = self._preprocess(frame_bgr)

    def has_background(self) -> bool:
        return self.background is not None

    def compute_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.background is None:
            raise RuntimeError("Background no capturado. Llamá a set_background() primero.")

        current = self._preprocess(frame_bgr)
        
        # Operación C++ directa sin conversiones pesadas
        diff = cv2.absdiff(self.background, current)
        diff_max = np.max(diff, axis=2)

        _, mask = cv2.threshold(diff_max, self.threshold, 255, cv2.THRESH_BINARY)
        return mask