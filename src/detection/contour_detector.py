"""
Detector de contornos con umbral de área de disparo para evitar sobre-segmentación.
"""

import cv2
import numpy as np
import supervision as sv
from scipy.ndimage import distance_transform_edt, label


class ContourDetector:
    def __init__(self, min_area: int = 800, max_single_area: int = 2200,
                 open_ksize: int = 3, close_ksize: int = 5,
                 roi: tuple[int, int, int, int] | None = None):
        """
        roi: (x1, y1, x2, y2) en píxeles — zona de la mesa a considerar.
             Todo lo que quede afuera se ignora (ej. cables, bordes de cámara).
             None = usa el recorte fijo de bordes anterior.
        """
        self.min_area = min_area
        self.max_single_area = max_single_area
        self.roi = roi
        self.update_kernels(open_ksize, close_ksize)

    def update_kernels(self, open_ksize: int, close_ksize: int) -> None:
        ok = max(1, open_ksize if open_ksize % 2 != 0 else open_ksize + 1)
        ck = max(1, close_ksize if close_ksize % 2 != 0 else close_ksize + 1)
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Permite actualizar la ROI en vivo (ej. desde un trackbar)."""
        self.roi = roi

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)

        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            roi_mask = np.zeros_like(closed)
            roi_mask[y1:y2, x1:x2] = 255
            closed = cv2.bitwise_and(closed, roi_mask)
        else:
            # fallback: recorte fijo de bordes (comportamiento anterior)
            closed[:60, :] = 0
            closed[-25:, :] = 0
            closed[:, :25] = 0
            closed[:, -25:] = 0

        return closed

    def _split_mask_watershed(self, mask_crop: np.ndarray) -> list[np.ndarray]:
        dist = distance_transform_edt(mask_crop)
        if dist.max() == 0:
            return [mask_crop]

        _, peaks = cv2.threshold(dist, 0.32 * dist.max(), 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)

        labeled_peaks, num_features = label(peaks)

        if num_features <= 1:
            return [mask_crop]

        markers = labeled_peaks.astype(np.int32) + 1
        unknown = mask_crop - peaks
        markers[unknown == 255] = 0

        crop_bgr = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
        cv2.watershed(crop_bgr, markers)

        sub_masks = []
        for feature_id in range(2, num_features + 2):
            sub_m = np.zeros_like(mask_crop)
            sub_m[markers == feature_id] = 255
            sub_masks.append(sub_m)

        return sub_masks

    def detect(self, clean_mask: np.ndarray) -> sv.Detections:
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xyxy_list = []
        mask_list = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            crop_mask = clean_mask[y:y+h, x:x+w]

            if area > self.max_single_area:
                sub_masks = self._split_mask_watershed(crop_mask)
            else:
                sub_masks = [crop_mask]

            for sub_m in sub_masks:
                sub_contours, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sub_c in sub_contours:
                    sub_area = cv2.contourArea(sub_c)
                    if sub_area < self.min_area:
                        continue

                    sx, sy, sw, sh = cv2.boundingRect(sub_c)
                    xyxy_list.append([x + sx, y + sy, x + sx + sw, y + sy + sh])

                    obj_mask = np.zeros(clean_mask.shape, dtype=bool)
                    cv2.drawContours(obj_mask[y:y+h, x:x+w].view(np.uint8), [sub_c], -1, 1, thickness=cv2.FILLED)
                    mask_list.append(obj_mask)

        if not xyxy_list:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            mask=np.array(mask_list, dtype=bool),
        )

    def get_contour_areas(self, clean_mask: np.ndarray) -> list[float]:
        """Devuelve el área de cada blob detectado, sin filtrar ni separar.
        Útil para calibrar min_area y max_single_area viendo valores reales."""
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.contourArea(c) for c in contours]