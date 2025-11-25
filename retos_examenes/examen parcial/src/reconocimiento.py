"""
reconocimiento.py
Funciones de reconocimiento por template matching y heurísticas optimizadas.
Incluye crops separados para rank y suit, y prints de depuración claros.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from .config import TEMPLATES_DIR, RANK_MATCH_THRESHOLD, SUIT_MATCH_THRESHOLD
from .utils import resize_keep_aspect

def load_templates():
    """
    Carga templates de ranks y suits desde data/templates/ranks y /suits.
    Devuelve dos diccionarios: ranks[name] = bin_img, suits[name] = bin_img
    """
    ranks_dir = Path(TEMPLATES_DIR) / "ranks"
    suits_dir = Path(TEMPLATES_DIR) / "suits"
    ranks = {}
    suits = {}

    if ranks_dir.exists():
        for p in sorted(ranks_dir.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ranks[p.stem] = b
    if suits_dir.exists():
        for p in sorted(suits_dir.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                suits[p.stem] = b
    print(f"[DEBUG] Templates cargados: {len(ranks)} ranks, {len(suits)} suits")
    return ranks, suits

def crop_rank(warped_img):
    """
    Crop del rank (esquina superior izquierda) y binarización.
    Ajusta coordenadas según tamaño de tu warp.
    """
    y1, y2, x1, x2 = 0, 200, 0, 100  # ejemplo, ajustar si tu warp es diferente
    crop = warped_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def crop_suit(warped_img):
    """
    Crop del suit (esquina inferior del rank) y binarización.
    Ajusta coordenadas según tamaño de tu warp.
    """
    y1, y2, x1, x2 = 0, 200, 0, 100   # ejemplo, ajustar según tu carta
    crop = warped_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def match_templates(crop_bin: np.ndarray, templates: dict) -> Tuple[str, float]:
    """
    matchTemplate con TM_CCOEFF_NORMED, devuelve (best_name, best_score)
    """
    best_name, best_score = "UNKNOWN", -1.0
    for name, tpl in templates.items():
        try:
            tpl_resized = cv2.resize(tpl, (crop_bin.shape[1], crop_bin.shape[0]))
        except Exception:
            continue
        res = cv2.matchTemplate(crop_bin, tpl_resized, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, _ = cv2.minMaxLoc(res)
        print(f"[DEBUG] Template {name} score: {maxval:.2f}")
        if maxval > best_score:
            best_score = maxval
            best_name = name
    return best_name, float(best_score)

def recognize_card_from_warp(warped_img, templates_rank, templates_suit):
    """
    Reconoce la carta dado el warp.
    Devuelve tuple: (rank, suit, rank_score, suit_score)
    """
    gray_rank = crop_rank(warped_img)
    gray_suit = crop_suit(warped_img)

    rank_name, rank_score = match_templates(gray_rank, templates_rank)
    suit_name, suit_score = match_templates(gray_suit, templates_suit)

    # Si score bajo, prueba versión invertida
    if rank_score < RANK_MATCH_THRESHOLD or suit_score < SUIT_MATCH_THRESHOLD:
        rank_inv = cv2.bitwise_not(gray_rank)
        suit_inv = cv2.bitwise_not(gray_suit)
        r2, rs2 = match_templates(rank_inv, templates_rank)
        s2, ss2 = match_templates(suit_inv, templates_suit)
        if rs2 > rank_score:
            rank_name, rank_score = r2, rs2
        if ss2 > suit_score:
            suit_name, suit_score = s2, ss2

    # Umbrales finales
    if rank_score < RANK_MATCH_THRESHOLD:
        rank_name = "UNKNOWN"
    if suit_score < SUIT_MATCH_THRESHOLD:
        suit_name = "UNKNOWN"

    print(f"[DEBUG] Resultado final: Rank={rank_name} ({rank_score:.2f}), Suit={suit_name} ({suit_score:.2f})")
    return rank_name, suit_name, rank_score, suit_score
