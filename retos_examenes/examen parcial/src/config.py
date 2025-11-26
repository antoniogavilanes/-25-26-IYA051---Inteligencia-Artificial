from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEMPLATES_DIR = Path("data/templates")
RESULT_WARPEDS = Path("results/warpeds")
RESULT_IMAGES = Path("results/detecciones")

RANK_MATCH_THRESHOLD = 0.05
SUIT_MATCH_THRESHOLD = 0.05

# Coordenadas de la esquina para crop de corner (si decides usar)
CORNER_CROP = (0, 100, 0, 70)
