"""
generate_templates.py
Genera templates de ranks y suits a partir de warps en data/processed.
Se pide al usuario el nombre de cada carta para generar templates precisos.
"""
import cv2
from pathlib import Path
from .config import PROCESSED_DIR, TEMPLATES_DIR
from .utils import ensure_dir, save_image

# Configuración de crops en el warp (ajustar según tus cartas)
RANK_CROP = (0, 200, 0, 100)    # y1,y2,x1,x2
SUIT_CROP = (0, 200, 0, 100)    # y1,y2,x1,x2

def crop_rank(warped_img):
    y1, y2, x1, x2 = RANK_CROP
    crop = warped_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def crop_suit(warped_img):
    y1, y2, x1, x2 = SUIT_CROP
    crop = warped_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img


def main():
    ensure_dir(TEMPLATES_DIR / "ranks")
    ensure_dir(TEMPLATES_DIR / "suits")

    warps = sorted((PROCESSED_DIR).glob("warped_*.jpg"))
    if not warps:
        print("No se encontraron warps en", PROCESSED_DIR)
        return

    for i, warp_path in enumerate(warps):
        print(f"\nProcesando warp: {warp_path.name}")
        img = cv2.imread(str(warp_path))
        if img is None:
            print(" ERROR leyendo imagen:", warp_path)
            continue

        rank_bin = crop_rank(img)
        suit_bin = crop_suit(img)

        # Mostrar al usuario para que indique nombre real
        cv2.imshow("Rank crop", rank_bin)
        cv2.imshow("Suit crop", suit_bin)
        cv2.waitKey(1)  # Necesario para renderizar

        rank_name = input("Ingresa el nombre del RANK (ej: A, 2, 10, K): ").strip()
        suit_name = input("Ingresa el nombre del SUIT (ej: corazon, picas, diamante, trebol): ").strip()

        # Guardar templates
        save_image(TEMPLATES_DIR / "ranks" / f"{rank_name}.png", rank_bin)
        save_image(TEMPLATES_DIR / "suits" / f"{suit_name}.png", suit_bin)

        cv2.destroyAllWindows()

    print("\nTemplates generados en:", TEMPLATES_DIR)

if __name__ == "__main__":
    main()
