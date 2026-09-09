"""
Detector de contornos optimizado para Celeron.
Ejecuta la separación por SciPy únicamente si el área de un contorno
supera un umbral alto (posibles piezas tocándose).
"""

import cv2
import numpy as np
import supervision as sv
from scipy.ndimage import distance_transform_edt, label


class ContourDetector:
    def __init__(self, min_area: int = 400, max_single_area: int = 3500, open_ksize: int = 3, close_ksize: int = 5):
        """
        min_area: área mínima para descartar ruido.
        max_single_area: área límite a partir de la cual se sospecha que hay 2+ piezas tocándose.
        """
        self.min_area = min_area
        self.max_single_area = max_single_area
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        # Morphological operations optimizadas por OpenCV
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        return closed

    def _separate_touching_objects(self, crop_mask: np.ndarray) -> np.ndarray:
        """Aplica SciPy solo sobre el recorte (crop) de la zona problemática, no sobre toda la pantalla."""
        dist = distance_transform_edt(crop_mask)
        if dist.max() == 0:
            return crop_mask

        _, foreground_seeds = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
        foreground_seeds = foreground_seeds.astype(np.uint8)

        labeled_seeds, num_features = label(foreground_seeds)
        if num_features <= 1:
            return crop_mask

        markers = labeled_seeds.astype(np.int32) + 1
        unknown = crop_mask - foreground_seeds
        markers[unknown == 255] = 0

        mask_color = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(mask_color, markers)
        
        separated = np.zeros_like(crop_mask)
        separated[markers > 1] = 255
        return separated

    def detect(self, clean_mask: np.ndarray) -> sv.Detections:
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xyxy_list = []
        mask_list = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Optimización: Solo invoca SciPy si el contorno es sospechosamente grande
            if area > self.max_single_area:
                crop = clean_mask[y:y+h, x:x+w]
                sep_crop = self._separate_touching_objects(crop)
                sub_contours, _ = cv2.findContours(sep_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for sub_c in sub_contours:
                    sub_area = cv2.contourArea(sub_c)
                    if sub_area < self.min_area:
                        continue
                    sx, sy, sw, sh = cv2.boundingRect(sub_c)
                    xyxy_list.append([x + sx, y + sy, x + sx + sw, y + sy + sh])

                    obj_mask = np.zeros(clean_mask.shape, dtype=bool)
                    cv2.drawContours(obj_mask[y:y+h, x:x+w].view(np.uint8), [sub_c], -1, 1, thickness=cv2.FILLED)
                    mask_list.append(obj_mask)
            else:
                xyxy_list.append([x, y, x + w, y + h])
                obj_mask = np.zeros(clean_mask.shape, dtype=bool)
                cv2.drawContours(obj_mask.view(np.uint8), [contour], -1, 1, thickness=cv2.FILLED)
                mask_list.append(obj_mask)

        if not xyxy_list:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            mask=np.array(mask_list, dtype=bool),
        )