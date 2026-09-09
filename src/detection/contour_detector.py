"""
Limpieza morfológica + separación de objetos tocándose usando SciPy (Distance Transform)
+ conversión a sv.Detections.
"""

import cv2
import numpy as np
import supervision as sv
from scipy.ndimage import distance_transform_edt, label


class ContourDetector:
    def __init__(self, min_area: int = 500, open_ksize: int = 5, close_ksize: int = 9, split_touching: bool = True):
        self.min_area = min_area
        self.split_touching = split_touching
        self.open_kernel = np.ones((open_ksize, open_ksize), np.uint8)
        self.close_kernel = np.ones((close_ksize, close_ksize), np.uint8)

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        return closed

    def _separate_touching_objects(self, mask: np.ndarray) -> np.ndarray:
        """
        Usa SciPy para calcular la transformada de distancia y separar objetos
        que están en contacto físico.
        """
        # Calcula la distancia al píxel de fondo más cercano
        dist = distance_transform_edt(mask)
        
        # Obtiene crestas de distancia (centros de los objetos)
        threshold_dist = 0.35 * dist.max() if dist.max() > 0 else 0
        _, foreground_seeds = cv2.threshold(dist, threshold_dist, 255, cv2.THRESH_BINARY)
        foreground_seeds = foreground_seeds.astype(np.uint8)

        # Etiqueta regiones con SciPy
        labeled_seeds, num_features = label(foreground_seeds)
        
        if num_features <= 1:
            return mask  # No se detectaron múltiples piezas unidas

        # Reconstruye la máscara separada por cuencas
        markers = labeled_seeds.astype(np.int32) + 1
        unknown = mask - foreground_seeds
        markers[unknown == 255] = 0

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(mask_color, markers)
        
        separated_mask = np.zeros_like(mask)
        separated_mask[markers > 1] = 255
        return separated_mask

    def detect(self, clean_mask: np.ndarray) -> sv.Detections:
        processed_mask = clean_mask
        if self.split_touching and np.any(clean_mask):
            processed_mask = self._separate_touching_objects(clean_mask)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xyxy_list = []
        mask_list = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
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