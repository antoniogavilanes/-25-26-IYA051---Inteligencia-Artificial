"""
captura.py
Funciones para capturar imágenes desde cámara o leer imágenes desde carpeta.
No hace nada mágico: te permite grabar frames en data/raw.
"""
from pathlib import Path
import cv2
from typing import Iterable, List
from .utils import ensure_dir
from .config import RAW_DIR

def capture_from_camera(save_dir: Path = RAW_DIR, device_id: int = 0, num_frames: int = 10):
    """
    Captura `num_frames` desde la cámara `device_id` y las guarda en save_dir.
    Útil para crear dataset de prueba con la cámara del portátil.
    """
    ensure_dir(save_dir)
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara {device_id}")
    saved = []
    i = 0
    print("Presiona SPACE para capturar un frame, ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura (espacio -> guardar)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            p = save_dir / f"cam_{i:03d}.jpg"
            cv2.imwrite(str(p), frame)
            saved.append(p)
            print("Guardado:", p)
            i += 1
            if i >= num_frames:
                break
    cap.release()
    cv2.destroyAllWindows()
    return saved

def list_images_in_folder(folder: Path) -> Iterable[Path]:
    """
    Devuelve rutas de imágenes dentro de `folder`.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in folder.iterdir():
        if p.suffix.lower() in exts:
            yield p
            
print("Leyendo imágenes desde:", RAW_DIR.resolve())

