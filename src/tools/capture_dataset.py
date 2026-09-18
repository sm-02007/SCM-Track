"""
Script standalone para captura y armado automático de dataset para IA.
Permite cambiar entre 10 clases usando teclas numéricas (1 al 9, y 0 para la décima)
y recortar individualmente cada objeto detectado mediante la tecla ESPACIO.
"""

import os
import sys
import time
import cv2

# Anclar la raíz del proyecto al PATH para resolver importaciones desde src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from capture.camera import Camera, load_background, save_background
from pipeline import InventoryPipeline

# -----------------------------------------------------------------------------
# CONSTANTES CONFIGURABLES
# -----------------------------------------------------------------------------
BACKGROUND_PATH = os.path.join(BASE_DIR, "data", "background.png")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")

# Tus 10 clases de accesorios de 1/2
CLASES = [
    "codo_90_1_2",
    "codo_45_1_2",
    "te_1_2",
    "curva_90_1_2",
    "union_doble_1_2",
    "cupla_1_2",
    "cupla_macho_1_2",
    "union_acople_1_2",
    "cupla_reductora_1_2",
    "valvula_esfera_exterior_1_2"
]

# Parámetros de segmentación sincronizados con main.py
BG_THRESHOLD = 25
MIN_AREA = 400
MAX_SINGLE_AREA = 8000
ROI = (1, 1, 640, 480)


def count_existing_images(class_name: str) -> int:
    """Cuenta cuántas imágenes existen en la carpeta de la clase actual."""
    class_folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_folder):
        return 0
    valid_exts = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(class_folder) if f.lower().endswith(valid_exts)]
    return len(files)


def ensure_directories():
    """Crea las 10 carpetas dentro de data/dataset/ si no existen."""
    for cls in CLASES:
        class_folder = os.path.join(DATASET_DIR, cls)
        os.makedirs(class_folder, exist_ok=True)


def main():
    ensure_directories()

    camera = None
    selected_class_idx = 0  # Inicia en la primera clase (codo_90_1_2)

    try:
        camera = Camera(index=1, width=640, height=480)
        pipeline = InventoryPipeline(
            bg_threshold=BG_THRESHOLD,
            min_area=MIN_AREA,
            max_single_area=MAX_SINGLE_AREA,
            roi=ROI,
        )

        bg = load_background(BACKGROUND_PATH)
        if bg is not None:
            pipeline.set_background(bg)
            print(f"[INFO] Fondo cargado exitosamente desde: {BACKGROUND_PATH}")
        else:
            print("[WARN] No se encontró fondo inicial en data/background.png. Presiona 'b' para capturar uno.")

        print("\n--- CAPTURA DE DATASET (10 CLASES) ---")
        print("Teclas asignadas por clase:")
        for i, cls_name in enumerate(CLASES):
            key_label = str(i + 1) if i < 9 else "0"
            print(f"  [{key_label}]: {cls_name}")
        print("\nControles:")
        print("  ESPACIO : Guardar recortado cada objeto detectado.")
        print("  b       : Recapturar background.")
        print("  q / ESC : Salir.\n")

        while True:
            frame = camera.read()
            if frame is None:
                continue

            clean_mask, annotated_frame, detections = pipeline.process(frame)

            current_class = CLASES[selected_class_idx]
            saved_in_class = count_existing_images(current_class)
            num_detections = len(detections) if detections is not None else 0

            # Información en pantalla
            text_line1 = f"Clase activa: {current_class} (Tecla 1-9, 0)"
            text_line2 = f"Fotos guardadas: {saved_in_class} | Detectados en frame: {num_detections}"

            # Sombra y texto
            cv2.putText(annotated_frame, text_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, text_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(annotated_frame, text_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, text_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Captura de Dataset - 10 Clases", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Salir con 'q' o ESC
            if key in (ord('q'), 27):
                break

            # Recapturar fondo con 'b'
            elif key == ord('b'):
                save_background(BACKGROUND_PATH, frame)
                pipeline.set_background(frame)
                print(f"[INFO] Nuevo fondo guardado y seteado en: {BACKGROUND_PATH}")

            # Teclas 1 al 9 -> Seleccionar clases 0 a 8
            elif ord('1') <= key <= ord('9'):
                selected_class_idx = key - ord('1')
                print(f"[CLASE CAMBIADA] Seleccionada: '{CLASES[selected_class_idx]}'")

            # Tecla 0 -> Seleccionar clase 9 (la 10ma)
            elif key == ord('0'):
                selected_class_idx = 9
                print(f"[CLASE CAMBIADA] Seleccionada: '{CLASES[selected_class_idx]}'")

            # Guardar recortes con ESPACIO
            elif key == ord(' '):
                if num_detections == 0:
                    print("[WARN] No hay objetos detectados en el frame para capturar.")
                    continue

                h_img, w_img = frame.shape[:2]
                saved_now = 0

                for idx, xyxy in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Clip a bordes de imagen
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_img, x2)
                    y2 = min(h_img, y2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    timestamp_ms = int(time.time() * 1000)
                    filename = f"{current_class}_{timestamp_ms}_{idx + 1}.png"
                    output_path = os.path.join(DATASET_DIR, current_class, filename)

                    cv2.imwrite(output_path, crop)
                    saved_now += 1

                print(f"[ÉXITO] Se guardaron {saved_now} imágenes en 'data/dataset/{current_class}/'")

    finally:
        if camera is not None and hasattr(camera, 'release'):
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()