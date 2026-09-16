"""
Extracción de features geométricos clásicos a partir de la silueta
individual de un objeto (máscara binaria de una sola pieza).
"""

import cv2
import numpy as np


def extract_features(object_mask: np.ndarray) -> dict | None:
    """
    Calcula features geométricos sobre la máscara de un objeto.
    """
    # Convertir a uint8 (0 y 255)
    mask_u8 = (object_mask.astype(np.uint8)) * 255

    # 1. Cierre morfológico suave (3x3) solo para eliminar micro-ruido de bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # 2. Buscar contornos externos e internos (RETR_CCOMP)
    contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]

    # Identificar el contorno externo principal (mayor área sin padre)
    main_idx = -1
    max_area = 0.0
    for idx, cnt in enumerate(contours):
        if hierarchy[idx][3] == -1:  # Sin padre = contorno externo
            area_cnt = cv2.contourArea(cnt)
            if area_cnt > max_area:
                max_area = area_cnt
                main_idx = idx

    if main_idx == -1 or max_area == 0:
        return None

    main_contour = contours[main_idx]
    perimeter = cv2.arcLength(main_contour, closed=True)

    # Circularidad
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Aspect Ratio
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = w / h if h > 0 else 0.0

    # Orientación
    if len(main_contour) >= 5:
        (_, _), (_, _), angle = cv2.fitEllipse(main_contour)
    else:
        angle = 0.0

    # 3. Detección robusta de huecos por diferencia de área sólida vs área real
    # Crear una imagen del objeto totalmente rellenado (sin huecos)
    filled_mask = np.zeros_like(clean_mask)
    cv2.drawContours(filled_mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    # La diferencia entre el objeto rellenado y la máscara real son los huecos
    holes_mask = cv2.bitwise_and(filled_mask, cv2.bitwise_not(clean_mask))

    # Contar cuántos huecos significativos existen dentro de la pieza
    hole_contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_holes = 0
    
    # Filtro: El hueco debe tener un área de al menos el 2% del tamaño total de la pieza
    min_hole_area = max_area * 0.02
    for h_cnt in hole_contours:
        if cv2.contourArea(h_cnt) >= min_hole_area:
            num_holes += 1

    return {
        "area": max_area,
        "perimeter": perimeter,
        "circularity": round(circularity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "orientation_deg": round(angle, 1),
        "num_holes": num_holes,
    }