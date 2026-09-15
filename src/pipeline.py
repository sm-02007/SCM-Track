"""
Pipeline con renderizado de siluetas/máscaras exactas y parámetros actualizables.
"""

import numpy as np
import supervision as sv

from detection.background_subtractor import BackgroundSubtractor
from detection.contour_detector import ContourDetector


class InventoryPipeline:
    def __init__(self, bg_threshold: int = 25, min_area: int = 400, max_single_area: int = 8000,
                 roi: tuple[int, int, int, int] | None = None):
        self.bg_subtractor = BackgroundSubtractor(threshold=bg_threshold)
        self.contour_detector = ContourDetector(min_area=min_area, max_single_area=max_single_area, roi=roi)

        # Anotadores visuales
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=2
        )
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.4
        )
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.TOP_CENTER
        )

    def update_parameters(self, bg_thresh: int, min_area: int, max_single_area: int, open_k: int, close_k: int) -> None:
        """Aplica los cambios de las trackbars en tiempo real."""
        self.bg_subtractor.threshold = bg_thresh
        self.contour_detector.min_area = min_area
        self.contour_detector.max_single_area = max_single_area
        self.contour_detector.update_kernels(open_k, close_k)

    def set_background(self, frame_bgr: np.ndarray) -> None:
        self.bg_subtractor.set_background(frame_bgr)

    def has_background(self) -> bool:
        return self.bg_subtractor.has_background()

    def process(self, frame_bgr: np.ndarray):
        raw_mask = self.bg_subtractor.compute_mask(frame_bgr)
        clean_mask = self.contour_detector.clean_mask(raw_mask)
        detections = self.contour_detector.detect(clean_mask)

        annotated = frame_bgr.copy()
        if len(detections) > 0:
            labels = [f"#{i+1}" for i in range(len(detections))]
            annotated = self.mask_annotator.annotate(annotated, detections)
            annotated = self.polygon_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        return clean_mask, annotated, detections