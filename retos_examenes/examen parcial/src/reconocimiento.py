"""
Reconocimiento de cartas usando templates por palo.
Acepta cualquier formato de imagen.
"""
import cv2
import numpy as np
from pathlib import Path

from .config import TEMPLATES_DIR

ACCEPTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def load_templates():
    """
    Carga todas las plantillas de cartas organizadas por palo.
    Devuelve un diccionario: {palo: {rank: imagen_bin}}
    """
    templates = {}
    templates_path = Path(TEMPLATES_DIR)
    for suit_dir in templates_path.iterdir():
        if suit_dir.is_dir():
            suit_name = suit_dir.name
            templates[suit_name] = {}
            for img_path in sorted(suit_dir.iterdir()):
                if img_path.suffix.lower() in ACCEPTED_EXTENSIONS:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    rank = img_path.stem.split("_")[0]  # ej: "A_warp_000.png" -> "A"
                    templates[suit_name][rank] = bin_img
    print(f"[DEBUG] Templates cargados por palo:")
    for s, ranks in templates.items():
        print(f"  {s}: {len(ranks)} cartas")
    return templates

def recognize_card_from_warp(warped_img, templates):
    """
    Reconoce la carta comparando con templates por palo.
    Devuelve: rank, suit, score
    """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, corner_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_score = float('inf')
    best_rank = "UNKNOWN"
    best_suit = "UNKNOWN"

    for suit, ranks in templates.items():
        for rank, tpl in ranks.items():
            tpl_resized = cv2.resize(tpl, (corner_bin.shape[1], corner_bin.shape[0]))
            res = cv2.matchTemplate(corner_bin, tpl_resized, cv2.TM_SQDIFF_NORMED)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < best_score:
                best_score = min_val
                best_rank = rank
                best_suit = suit

    return best_rank, best_suit, best_score
