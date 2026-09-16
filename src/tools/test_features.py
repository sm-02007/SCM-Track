"""
Script de prueba rápida para geometric_features.py.
Muestra video en vivo, y con la tecla ESPACIO imprime en consola
los features de cada objeto detectado en ese frame.
"""

import os
import sys
import cv2

# Anclar la raíz del proyecto al PATH para resolver importaciones correctamente
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from capture.camera import Camera, load_background
from pipeline import InventoryPipeline
from features.geometric_features import extract_features

BACKGROUND_PATH = os.path.join(BASE_DIR, "data", "background.png")

# Parámetros sincronizados con main.py
BG_THRESHOLD = 25
MIN_AREA = 400
MAX_SINGLE_AREA = 8000
ROI = (23, 113, 458, 480)


def main():
    camera = None
    try:
        camera = Camera(index=1, width=640, height=480)
        pipeline = InventoryPipeline(
            bg_threshold=BG_THRESHOLD,
            min_area=MIN_AREA,
            max_single_area=MAX_SINGLE_AREA,
            roi=ROI,
        )

        bg = load_background(BACKGROUND_PATH)
        if bg is None:
            print(f"[ERROR] No se pudo cargar la imagen de fondo desde:\n  {BACKGROUND_PATH}")
            return

        pipeline.set_background(bg)
        print("\n--- TEST DE FEATURES GEOMÉTRICOS ---")
        print("Presiona 'ESPACIO' para imprimir métricas del frame actual en consola.")
        print("Presiona 'q' o 'ESC' para salir.\n")

        while True:
            frame = camera.read()
            if frame is None:
                print("[WARN] No se pudo obtener frame de la cámara.")
                continue

            clean_mask, annotated_frame, detections = pipeline.process(frame)

            #cv2.imshow("Test Features - Video en Vivo", annotated_frame)
            cv2.imshow("Test Features - Mascara Binaria", clean_mask)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            # Extraer e imprimir métricas con ESPACIO
            elif key == ord(' '):
                num_detections = len(detections) if detections is not None else 0
                print(f"\n==================== OBJETOS DETECTADOS: {num_detections} ====================")

                if num_detections == 0:
                    print("No se detectó ningún objeto en la escena.")
                    continue

                for idx in range(num_detections):
                    # Extraer la máscara booleana/binaria de la detección individual de Supervision
                    det_mask = detections.mask[idx]

                    # Pasar la máscara directamente a extract_features (tal como fue diseñada)
                    feats = extract_features(det_mask)
                    
                    print(f"\n[Objeto #{idx + 1}]")
                    if feats is None:
                        print("  [WARN] No se pudo calcular features (contorno inválido o área 0).")
                        continue

                    for key_feat, val_feat in feats.items():
                        if isinstance(val_feat, float):
                            print(f"  {key_feat:<15}: {val_feat:.3f}")
                        else:
                            print(f"  {key_feat:<15}: {val_feat}")

    finally:
        if camera is not None and hasattr(camera, 'release'):
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()