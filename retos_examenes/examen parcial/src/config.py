"""
config.py
Parámetros globales y rutas del proyecto.
Ajusta estos valores según tus pruebas.
"""
from pathlib import Path
import numpy as np

# Ruta base del proyecto (ajústala si quieres hardcodearla)
BASE_DIR = Path(__file__).resolve().parents[1]  # ../ (carpeta examen parcial)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEMPLATES_DIR = DATA_DIR / "templates"
TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = BASE_DIR / "results"
DETECTIONS_DIR = RESULTS_DIR / "detecciones"
LOGS_FILE = RESULTS_DIR / "logs.txt"

# Warping / tamaño estándar de carta (mantener proporción ~ 5:7)
WARP_W = 500
WARP_H = 700

# Segmentación color (HSV) - rango para "verde" del tapete (ajustar con calibración)
# Valores por defecto: H: 35-85, S: 40-255, V: 40-255
GREEN_LOWER = np.array([35, 40, 40], dtype=int)
GREEN_UPPER = np.array([85, 255, 255], dtype=int)

# Parámetros morfológicos y filtrado
MORPH_KERNEL = (5, 5)
MORPH_ITERATIONS_OPEN = 1
MORPH_ITERATIONS_CLOSE = 2
AREA_MIN = 2000  # área mínima para considerar un contorno (ajustar según resolución)

# Matching / reconocimiento
RANK_MATCH_THRESHOLD = 0.05
SUIT_MATCH_THRESHOLD = 0.05
CORNER_CROP = (0, 200, 0, 100) # y1,y2,x1,x2

# también podrías usar múltiples crops: top-left, top-right, bottom-left, bottom-right

# Otras opciones
SHOW_DEBUG_WINDOWS = True  # True para ver ventanas con resultados intermedios
SAVE_WARPED = True         # True para guardar cada carta warp en processed/
