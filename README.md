# Detección de Billetes con YOLOv8
**Proyecto de Innovación — Maestría en IA Aplicada**

Modelo de visión artificial que detecta billetes colombianos de $2.000, $10.000 y $50.000
y calcula el total del dinero visible en escena usando una cámara en tiempo real.

---

## Contenido del Repositorio

| Archivo / Carpeta | Descripción |
|---|---|
| `best.pt` | Modelo entrenado (YOLOv8m) — listo para usar |
| `Billetes_Colombia_YOLOv8_FINAL.ipynb` | Notebook principal: entrenamiento en Google Colab |
| `Billetes_MacBook_Local.ipynb` | Notebook para correr el modelo en MacBook con cámara |
| `Video_test.mp4` | **Evidencia de prueba** — detección en tiempo real con resultados |
| `bill_train.v1i.yolov8/` | Dataset: imágenes etiquetadas en formato YOLOv8 |

---

## Evidencia de Prueba

El archivo `Video_test.mp4` contiene la demostración del modelo funcionando en tiempo real
sobre una cámara MacBook, obteniendo **muy buenos resultados** en la detección y clasificación
de los tres billetes:

- Detección correcta de billetes de $2.000, $10.000 y $50.000
- Cálculo preciso del total del dinero visible en escena
- Funcionamiento fluido a ~15 FPS en MacBook con chip Apple Silicon
- Robustez ante variaciones de iluminación, ángulo y estado del billete

> El video fue grabado directamente desde el notebook `Billetes_MacBook_Local.ipynb`
> usando el modelo `best.pt` entrenado en Google Colab.

---

## Opción 1 — Usar el modelo ya entrenado en MacBook

**Requisito:** tener Python 3.11 instalado.

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv billetes_env
source billetes_env/bin/activate
pip install ultralytics opencv-python jupyter matplotlib

# 3. Lanzar Jupyter
jupyter notebook
```

Abrir `Billetes_MacBook_Local.ipynb` y ejecutar los bloques en orden:
- **Bloque 1** — carga `best.pt` y verifica la cámara
- **Bloque 2** — video en tiempo real (`q` para salir, `s` para guardar captura)
- **Bloque 3** — modo foto (resultado dentro del notebook)

> El modelo usa automáticamente la GPU Apple Silicon (MPS) si está disponible.

---

## Opción 2 — Reentrenar el modelo en Google Colab

1. Subir la carpeta `bill_train.v1i.yolov8/` a Google Drive en la ruta:
   ```
   Mi unidad/Proyecto_innovacion_2/Tarea_2/dataset/
   ```
2. Abrir `Billetes_Colombia_YOLOv8_FINAL.ipynb` en Google Colab
3. Activar GPU: `Entorno de ejecución → Cambiar tipo → GPU T4`
4. Ejecutar todos los bloques en orden

El notebook entrena el modelo, evalúa las métricas, valida con la cámara del PC
y exporta el modelo a Drive en formato `.pt` y `.onnx`.

---

## Parámetros ajustables (Bloque 1 del notebook Mac)

```python
RUTA_MODELO   = './best.pt'  # ruta al modelo
INDICE_CAMARA = 0            # 0 = cámara integrada, 1 = USB externa
CONFIANZA     = 0.45         # bajar a 0.30 si no detecta, subir a 0.60 si hay errores
IOU_THRESH    = 0.30         # bajar si aparecen cajas duplicadas del mismo billete
```

---

## Detalles Técnicos

- **Arquitectura:** YOLOv8m con Transfer Learning desde COCO
- **Clases:** `2000_pesos` · `10000_pesos` · `50000_pesos`
- **Dataset:** ~300 imágenes etiquetadas con Roboflow
- **Augmentación:** perspectiva, cizallamiento, mosaic, copy_paste, mixup
- **Optimizador:** SGD lr=0.005, weight_decay=0.001
- **Métricas:** mAP50 = 0.995 en validación
- **Despliegue:** MacBook (MPS) · Google Colab (T4) · Jetson Nano Orin (TensorRT)

---

## Dependencias

```
ultralytics >= 8.4
opencv-python >= 4.8
torch >= 2.0
matplotlib >= 3.7
pyyaml >= 6.0
```
