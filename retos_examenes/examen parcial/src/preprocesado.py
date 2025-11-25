"""
preprocesado.py
Operaciones de preprocesado: máscara por color, morfología, umbral, etc.
"""
import cv2
import numpy as np
from .config import GREEN_LOWER, GREEN_UPPER, MORPH_KERNEL, MORPH_ITERATIONS_OPEN, MORPH_ITERATIONS_CLOSE
from typing import Tuple

def mask_green_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Devuelve la máscara binaria de la región que NO es verde (invertida),
    es decir, donde pueden estar las cartas.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = GREEN_LOWER.astype(int)
    upper = GREEN_UPPER.astype(int)
    mask_green = cv2.inRange(hsv, lower, upper)
    mask_not_green = cv2.bitwise_not(mask_green)
    return mask_not_green

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Aplica operaciones morfológicas para limpiar la máscara.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS_OPEN)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE)
    return closed

def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binariza la imagen en escala de grises con adaptive threshold.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # Parámetros por defecto razonables; ajustar si hace falta
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    return bin_img
