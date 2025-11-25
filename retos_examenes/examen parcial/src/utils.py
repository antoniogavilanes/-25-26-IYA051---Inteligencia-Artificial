"""
utils.py
Funciones auxiliares: manejo de paths, warp, orden de puntos, dibujo simple.
"""
from typing import Tuple
import cv2
import numpy as np
from pathlib import Path
from .config import WARP_W, WARP_H
import os

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena pts (4x2) a [tl, tr, br, bl]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_card(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Aplica perspectiva para obtener vista top-down de la carta.
    `pts` expected shape (4,1,2) or (4,2)
    """
    if pts.shape[1] == 1:
        pts = pts.reshape(4, 2)
    rect = order_points(pts.astype("float32"))
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (WARP_W, WARP_H))
    return warped

def resize_keep_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def draw_bbox_with_label(img, approx, label: str, color=(0,255,0)):
    rect = cv2.boundingRect(approx)
    x,y,w,h = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def save_image(path: Path, img):
    ensure_parent = path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
