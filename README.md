# Inventario CV — MVP (visión clásica)

## Instalación
```
pip install -r requirements.txt
```

## Uso
```
cd src
python main.py
```
- `b` → capturar background (hacerlo con la mesa vacía)
- `q` → salir

## Arquitectura
- `capture/camera.py` → cámara + guardado/carga de background
- `detection/background_subtractor.py` → frame vs background → máscara binaria
- `detection/contour_detector.py` → limpieza morfológica + contornos → `sv.Detections`
- `pipeline.py` → orquesta todo, sin saber de cámara/UI
- `main.py` → loop, ventanas, teclas

## Ajustes disponibles (en `InventoryPipeline.__init__`)
- `bg_threshold`: sensibilidad a diferencias (subir si detecta ruido de la madera)
- `min_area`: área mínima en px para contar como objeto (subir si detecta ruido pequeño)

## Pendiente (no implementado en este MVP)
- Separación de objetos que se tocan (watershed + distance transform)
- Reconocimiento de tipo de accesorio
- Integración con modelo de IA
# SCM-Track
