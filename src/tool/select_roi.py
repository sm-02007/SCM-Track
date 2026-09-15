"""
Herramienta standalone para elegir la ROI (zona útil de la mesa) con el mouse.

Uso:
  python tools/select_roi.py

Instrucciones en pantalla:
  - Click y arrastrá para dibujar el rectángulo sobre la mesa.
  - 'r' reinicia la selección.
  - Enter/Espacio confirma y muestra las coordenadas en la terminal.
  - 'q' sale sin guardar.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture.camera import Camera


def main():
    camera = Camera(index=1)

    print("Presioná cualquier tecla en la ventana de video para tomar un frame de referencia.")
    frame = camera.read()
    camera.release()

    # cv2.selectROI abre una ventana interactiva: click+arrastre, Enter para confirmar
    print("Dibujá el rectángulo sobre la zona útil de la mesa (click+arrastre), luego ENTER.")
    x, y, w, h = cv2.selectROI("Seleccionar ROI - ENTER para confirmar", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("No se seleccionó ninguna zona.")
        return

    x1, y1, x2, y2 = x, y, x + w, y + h
    print("\nROI seleccionada:")
    print(f"roi=({x1}, {y1}, {x2}, {y2})")
    print("\nPegá esto donde instancies ContourDetector en pipeline.py")


if __name__ == "__main__":
    main()