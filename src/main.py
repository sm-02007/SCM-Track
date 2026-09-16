"""
Interfaz gráfica idéntica al diseño de presentación.
Incluye esquinas redondeadas en el contenedor de video, recuadro de conteo y botones.
"""

import os
import sys
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import cv2

# Anclar la raíz del proyecto al PATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from capture.camera import Camera, load_background, save_background
from pipeline import InventoryPipeline

BACKGROUND_PATH = os.path.join(BASE_DIR, "data", "background.png")

# Parámetros de segmentación
BG_THRESHOLD = 25
MIN_AREA = 400
MAX_SINGLE_AREA = 10000
OPEN_KSIZE = 3
CLOSE_KSIZE = 11
ROI = (1, 1, 640, 480)


class RoundedButton(tk.Canvas):
    """Botón estilo cápsula con bordes redondeados."""

    def __init__(self, parent, text, command=None, bg_color="#1200A3", fg_color="#FFFFFF", font=("Arial", 18, "bold"), **kwargs):
        super().__init__(parent, bg="#FFFFFF", highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.text = text
        self.font = font

        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        r = h // 2

        self.create_arc((0, 0, h, h), start=90, extent=180, fill=self.bg_color, outline=self.bg_color)
        self.create_arc((w - h, 0, w, h), start=-90, extent=180, fill=self.bg_color, outline=self.bg_color)
        self.create_rectangle((r, 0, w - r, h), fill=self.bg_color, outline=self.bg_color)
        self.create_text(w // 2, h // 2, text=self.text, fill=self.fg_color, font=self.font)


def add_corners_to_image(pil_img, radius=25):
    """Aplica un recorte de esquinas redondeadas sobre una imagen PIL."""
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, pil_img.size[0], pil_img.size[1]), radius=radius, fill=255)
    
    output = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    output.paste(pil_img, (0, 0), mask)
    return output


class AppUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Conteo e Inventario CV")
        self.root.geometry("1280x720")
        self.root.configure(bg="#FFFFFF")

        # Inicialización de cámara y pipeline
        self.camera = Camera(index=1, width=640, height=480)
        self.pipeline = InventoryPipeline(
            bg_threshold=BG_THRESHOLD,
            min_area=MIN_AREA,
            max_single_area=MAX_SINGLE_AREA,
            roi=ROI,
        )
        self.pipeline.contour_detector.update_kernels(OPEN_KSIZE, CLOSE_KSIZE)

        existing_bg = load_background(BACKGROUND_PATH)
        if existing_bg is not None:
            self.pipeline.set_background(existing_bg)

        self._build_gui()
        self.root.bind("<b>", self.capture_background)
        self.update_video()

    def _build_gui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # SECCIÓN IZQUIERDA: CONTENEDOR DE VIDEO CON ESQUINAS REDONDEADAS
        # ---------------------------------------------------------------------
        self.left_frame = tk.Frame(self.root, bg="#FFFFFF")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=30, pady=30)

        self.video_container = tk.Frame(self.left_frame, bg="#FFFFFF")
        self.video_container.pack(fill="both", expand=True)

        self.video_label = tk.Label(self.video_container, bg="#FFFFFF")
        self.video_label.pack(fill="both", expand=True)

        # ---------------------------------------------------------------------
        # SECCIÓN DERECHA: PANEL DE CONTROL
        # ---------------------------------------------------------------------
        self.right_frame = tk.Frame(self.root, bg="#FFFFFF")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=30)

        # Título "CANTIDAD"
        self.lbl_cantidad = tk.Label(
            self.right_frame,
            text="CANTIDAD",
            font=("Montserrat", 38, "bold"),
            fg="#F16522",
            bg="#FFFFFF"
        )
        self.lbl_cantidad.pack(anchor="center", pady=(10, 20))

        # Recuadro con el contador
        self.rect_frame = tk.Frame(
            self.right_frame,
            bg="#FFFFFF",
            bd=2,
            relief="solid",
            width=360,
            height=180
        )
        self.rect_frame.pack(anchor="center", pady=(0, 40))
        self.rect_frame.pack_propagate(False)

        self.lbl_counter = tk.Label(
            self.rect_frame,
            text="0",
            font=("Open Sans", 72, "bold"),
            fg="#000000",
            bg="#FFFFFF"
        )
        self.lbl_counter.pack(expand=True)

        # Botones inferiores redondeados
        self.btn_salida = RoundedButton(
            self.right_frame,
            text="Salida",
            command=self.on_exit,
            width=400,
            height=75
        )
        self.btn_salida.pack(anchor="center", pady=10)

        self.btn_config = RoundedButton(
            self.right_frame,
            text="Configuración",
            command=self.on_config,
            width=400,
            height=75
        )
        self.btn_config.pack(anchor="center", pady=10)

        self.btn_reset = RoundedButton(
            self.right_frame,
            text="Reset",
            command=self.on_reset,
            width=400,
            height=75
        )
        self.btn_reset.pack(anchor="center", pady=10)

    def capture_background(self, event=None):
        frame = self.camera.read()
        if frame is not None:
            self.pipeline.set_background(frame)
            save_background(BACKGROUND_PATH, frame)

    def update_video(self):
        frame = self.camera.read()

        if frame is not None:
            if self.pipeline.has_background():
                mask, annotated, detections = self.pipeline.process(frame)
                num_objetos = len(detections) if detections is not None else 0
                self.lbl_counter.config(text=str(num_objetos))
                display_frame = annotated
            else:
                self.lbl_counter.config(text="--")
                cv2.putText(
                    frame, "SIN FONDO - Presiona 'b' o RESET",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                display_frame = frame

            # Redimensionar y aplicar borde suave redondeado al video
            vw = self.video_container.winfo_width()
            vh = self.video_container.winfo_height()

            if vw > 10 and vh > 10:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (vw, vh))
                
                # Convertir a PIL y redondear bordes con radio 25
                pil_img = Image.fromarray(resized_frame)
                rounded_img = add_corners_to_image(pil_img, radius=25)
                
                imgtk = ImageTk.PhotoImage(image=rounded_img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_video)

    def on_exit(self):
        if hasattr(self.camera, 'release'):
            self.camera.release()
        self.root.destroy()

    def on_config(self):
        print("[ACCION] Configuración presionado.")

    def on_reset(self):
        self.capture_background()


def main():
    root = tk.Tk()
    app = AppUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()


if __name__ == "__main__":
    main()