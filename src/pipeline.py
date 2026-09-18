"""
Pipeline unificado y optimizado para hardware liviano.
"""

import numpy as np
import supervision as sv

from detection.background_subtractor import BackgroundSubtractor
from detection.contour_detector import ContourDetector


class InventoryPipeline:
    def __init__(self, bg_threshold: int = 25, min_area: int = 400):
        self.bg_subtractor = BackgroundSubtractor(threshold=bg_threshold)
        self.contour_detector = ContourDetector(min_area=min_area)

        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    def set_background(self, frame_bgr: np.ndarray) -> None:
        self.bg_subtractor.set_background(frame_bgr)

    def has_background(self) -> bool:
        return self.bg_subtractor.has_background()

    def process(self, frame_bgr: np.ndarray):
        raw_mask = self.bg_subtractor.compute_mask(frame_bgr)
        clean_mask = self.contour_detector.clean_mask(raw_mask)
        
        # Pasa directamente clean_mask
        detections = self.contour_detector.detect(clean_mask)

        annotated = frame_bgr.copy()
        if len(detections) > 0:
            labels = [f"#{i+1}" for i in range(len(detections))]
            annotated = self.mask_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        return clean_mask, annotated, detections