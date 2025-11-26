import cv2
from pathlib import Path

def ensure_dir(path: Path):
    """Crea el directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)

def save_image(path: Path, img):
    """Guarda la imagen asegurando que el directorio exista."""
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

def draw_bbox_with_label(img, approx, label):
    """Dibuja un polígono y un label sobre la imagen."""
    cv2.polylines(img, [approx], True, (0, 255, 0), 2)
    # Pone el texto ligeramente arriba del primer punto del polígono
    x, y = approx[0][0], approx[0][1]
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
