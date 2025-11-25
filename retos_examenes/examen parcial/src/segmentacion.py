"""
segmentacion.py
Detección de contornos y extracción/wrap de cartas.
"""
import cv2
import numpy as np
from typing import List, Dict
from .preprocesado import mask_green_hsv, clean_mask
from .utils import warp_card, ensure_dir, save_image
from .config import AREA_MIN, PROCESSED_DIR, SAVE_WARPED, WARP_W, WARP_H
from pathlib import Path

def find_card_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append(approx)
    return candidates

def extract_warped_cards(img_bgr, approximations: List[np.ndarray], start_idx: int = 0) -> (List[Dict], int):
    ensure_dir(PROCESSED_DIR)
    results = []
    idx = start_idx
    for approx in approximations:
        warped = warp_card(img_bgr, approx)
        if SAVE_WARPED:
            save_image(PROCESSED_DIR / f"warped_{idx:03d}.jpg", warped)
        results.append({"approx": approx, "warped": warped})
        idx += 1
    return results, idx

def detect_cards_in_image(img_bgr, img_name: str = "", start_idx: int = 0):
    mask = mask_green_hsv(img_bgr)
    mask_clean = clean_mask(mask)
    approximations = find_card_contours(mask_clean)
    warps, new_idx = extract_warped_cards(img_bgr, approximations, start_idx=start_idx)
    return warps, mask_clean, new_idx
