"""
Genera templates automáticamente desde warpeds.
Muestra la carta recortada y permite asignar número y palo.
Acepta cualquier formato de imagen en RAW_DIR.
"""
import cv2
from pathlib import Path
from src.segmentacion import detect_cards_in_image

RAW_DIR = Path("data/raw")
TEMPLATES_DIR = Path("data/templates")
RESULT_WARPEDS = Path("results/warpeds")

ACCEPTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def main():
    for img_path in sorted(RAW_DIR.glob("*.*")):
        if img_path.suffix.lower() not in ACCEPTED_EXTENSIONS:
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        warps, _, _ = detect_cards_in_image(img_bgr, img_name=img_path.stem)
        for w in warps:
            # Mostrar warp para identificar número y palo
            cv2.imshow("Warped Card", w["warped"])
            cv2.waitKey(1)  # Pequeño delay para mostrar la ventana

            # Input de número y palo
            rank = input("Introduce el número o letra (A,2,...,K): ").strip()
            suit = input("Introduce el palo (corazones/picas/diamantes/trebol): ").strip()

            # Crear carpeta del palo
            suit_dir = TEMPLATES_DIR / suit
            suit_dir.mkdir(parents=True, exist_ok=True)

            # Guardar la imagen con nombre: RANK_WARP_XXX.png
            existing = list(suit_dir.glob(f"{rank}_warp_*.png"))
            idx = len(existing)
            save_path = suit_dir / f"{rank}_warp_{idx:03d}.png"
            cv2.imwrite(str(save_path), w["warped"])
            print(f"Guardado: {save_path}")

            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
