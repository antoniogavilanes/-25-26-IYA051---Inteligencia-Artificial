"""
Genera templates automáticos usando corners extraídos de los warps.
"""
import cv2
from pathlib import Path
from src.segmentacion import detect_cards_in_image
from src.config import TEMPLATES_DIR, RAW_DIR, CORNER_CROP

ACCEPTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def main():
    raw_path = Path(RAW_DIR)

    for img_path in sorted(raw_path.glob("*.*")):
        if img_path.suffix.lower() not in ACCEPTED_EXTENSIONS:
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        warps, _, _ = detect_cards_in_image(img_bgr, img_name=img_path.stem)

        for w in warps:
            corner = w['corner']
            cv2.imshow("Corner", corner)
            cv2.waitKey(1)

            # Input de rank y palo
            rank = input("Introduce el número o letra (A,2,...,K): ").strip().upper()
            suit = input("Introduce el palo (corazones/picas/diamantes/trebol): ").strip().lower()

            if suit not in {"corazones", "picas", "diamantes", "trebol"}:
                print("Palo inválido, se ignora esta carta")
                continue

            suit_dir = TEMPLATES_DIR / suit
            suit_dir.mkdir(parents=True, exist_ok=True)

            existing = list(suit_dir.glob(f"{rank}_warp_*.png"))
            idx = len(existing)
            save_path = suit_dir / f"{rank}_warp_{idx:03d}.png"
            cv2.imwrite(str(save_path), corner)
            print(f"Guardado: {save_path}")

            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
