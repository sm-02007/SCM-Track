"""
Loop principal con selector automático de cámara.

Teclas:
  b -> capturar/actualizar background
  q -> salir
"""

import os
import cv2

from capture.camera import Camera, save_background, load_background
from pipeline import InventoryPipeline

BACKGROUND_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "background.png")


def main():
    # index=None activa la búsqueda automática del primer puerto funcional
    camera = Camera(index=1)
    pipeline = InventoryPipeline(bg_threshold=25, min_area=500)

    # Si ya había un background guardado de una corrida anterior, lo cargamos
    existing_bg = load_background(BACKGROUND_PATH)
    if existing_bg is not None:
        pipeline.set_background(existing_bg)
        print("[Main] Background previo cargado desde disco.")

    print("\n-----------------------------------------------------")
    print("INSTRUCCIONES:")
    print("  Presioná 'b' para capturar un nuevo background.")
    print("  Presioná 'q' para salir.")
    print("-----------------------------------------------------\n")

    try:
        while True:
            frame = camera.read()

            cv2.imshow("Video en vivo", frame)

            if pipeline.has_background():
                mask, annotated, detections = pipeline.process(frame)
                cv2.imshow("Mascara Binaria", mask)
                
                # Renderiza conteo sobre la ventana anotada
                cv2.putText(
                    annotated, f"Objetos detectados: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Detecciones (Supervision)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("b"):
                pipeline.set_background(frame)
                save_background(BACKGROUND_PATH, frame)
                print("[Main] Nuevo background capturado y guardado.")
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()