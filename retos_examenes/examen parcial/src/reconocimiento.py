import cv2
import numpy as np
from pathlib import Path
from src.config import TEMPLATES_DIR, RANK_MATCH_THRESHOLD

ACCEPTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def load_templates():
    templates = {}
    for suit_dir in Path(TEMPLATES_DIR).iterdir():
        if suit_dir.is_dir():
            suit_name = suit_dir.name
            templates[suit_name] = {}
            for img_path in sorted(suit_dir.iterdir()):
                if img_path.suffix.lower() in ACCEPTED_EXTENSIONS:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    rank = img_path.stem.split("_")[0]
                    templates[suit_name][rank] = bin_img
    return templates


def recognize_card_from_warp(corner_img, templates):
    gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
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

    if best_score > RANK_MATCH_THRESHOLD:
        return "UNKNOWN", "UNKNOWN", best_score

    return best_rank, best_suit, best_score
