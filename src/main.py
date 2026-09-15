"""
Loop principal. Sin trackbars: los parámetros se ajustan editando las
constantes de abajo directamente en el código.
"""

import os
import cv2

from capture.camera import Camera, save_background, load_background
from pipeline import InventoryPipeline

BACKGROUND_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "background.png")

# ------------------------------------------------------------------
# PARAMETROS DE CALIBRACION - editar estos valores a mano y reiniciar
# ------------------------------------------------------------------
BG_THRESHOLD = 25          # sensibilidad al fondo (subir si detecta ruido de la madera)
MIN_AREA = 400             # área mínima en px para contar como objeto (ignora ruido chico)
MAX_SINGLE_AREA = 10000     # área máxima para 1 sola pieza (por encima, intenta separar en 2+)
OPEN_KSIZE = 3             # limpieza de puntos sueltos
CLOSE_KSIZE = 11            # relleno de huecos internos
ROI = (21, 112, 497, 449)  # zona útil de la mesa (x1, y1, x2, y2)
# ------------------------------------------------------------------


def main():
    camera = Camera(index=1, width=640, height=480)
    pipeline = InventoryPipeline(
        bg_threshold=BG_THRESHOLD,
        min_area=MIN_AREA,
        max_single_area=MAX_SINGLE_AREA,
        roi=ROI,
    )
    pipeline.contour_detector.update_kernels(OPEN_KSIZE, CLOSE_KSIZE)

    existing_bg = load_background(BACKGROUND_PATH)
    if existing_bg is not None:
        pipeline.set_background(existing_bg)
        print("[Main] Background previo cargado.")

    print("\n-----------------------------------------------------")
    print("  'b' -> capturar/actualizar background (mesa vacía)")
    print("  'q' -> salir")
    print("  Para ajustar sensibilidad, editá las constantes al inicio de main.py")
    print("-----------------------------------------------------\n")

    while True:
        frame = camera.read()

        if pipeline.has_background():
            mask, annotated, detections = pipeline.process(frame)

            cv2.putText(
                annotated, f"Objetos: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            areas = pipeline.contour_detector.get_contour_areas(mask)
            if areas:
                areas_txt = ", ".join(f"{int(a)}" for a in areas if a > MIN_AREA)
                cv2.putText(
                    annotated, f"Areas: {areas_txt}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

            cv2.imshow("Mascara Binaria", mask)
            cv2.imshow("Siluetas (Supervision)", annotated)
        else:
            cv2.imshow("Video en vivo (Sin Fondo)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            pipeline.set_background(frame)
            save_background(BACKGROUND_PATH, frame)
            print("[Main] Nuevo fondo capturado.")
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()