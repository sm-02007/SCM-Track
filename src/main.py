"""
Loop principal con renderizado liviano (desactiva ventanas innecesarias para ahorrar CPU).
"""

import os
import cv2

from capture.camera import Camera, save_background, load_background
from pipeline import InventoryPipeline

BACKGROUND_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "background.png")


def main():
    camera = Camera(index=1, width=640, height=480)
    pipeline = InventoryPipeline(bg_threshold=25, min_area=400)

    existing_bg = load_background(BACKGROUND_PATH)
    if existing_bg is not None:
        pipeline.set_background(existing_bg)
        print("[Main] Background previo cargado.")

    print("Presioná 'b' para capturar background, 'q' para salir.")

    while True:
        frame = camera.read()

        if pipeline.has_background():
            mask, annotated, detections = pipeline.process(frame)
            
            # Solo dibujamos texto si hay cambios
            cv2.putText(
                annotated, f"Objetos: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            
            # Mostramos únicamente 2 ventanas para reducir overhead de GUI en Linux
            cv2.imshow("Mascara", mask)
            cv2.imshow("Detecciones (Supervision)", annotated)
        else:
            cv2.imshow("Video en vivo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            pipeline.set_background(frame)
            save_background(BACKGROUND_PATH, frame)
            print("[Main] Background actualizado.")
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()