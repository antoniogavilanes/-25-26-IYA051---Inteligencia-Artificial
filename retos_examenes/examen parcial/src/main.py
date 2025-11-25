"""
main.py
Pipeline completo: segmentación -> warp -> reconocimiento.
"""
import cv2
from pathlib import Path
from src.segmentacion import detect_cards_in_image
from src.reconocimiento import load_templates, recognize_card_from_warp
from src.utils import draw_bbox_with_label, save_image, ensure_dir
from src.config import RAW_DIR, RESULTS_DIR

def main():
    raw_path = Path(RAW_DIR)
    results_path = Path(RESULTS_DIR) / "detecciones"
    ensure_dir(results_path)

    print(f"Leyendo imágenes desde: {raw_path}")

    templates_rank, templates_suit = load_templates()
    if not templates_rank or not templates_suit:
        print("ERROR: No se encontraron templates de ranks o suits")
        return

    # Contador global para warpeds
    warp_idx = 0

    for img_path in sorted(raw_path.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        print(f"Procesando: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"ERROR: No se pudo leer la imagen {img_path.name}")
            continue

        # Detectar cartas y extraer warpeds
        warps, mask_clean, warp_idx = detect_cards_in_image(img_bgr, img_name=img_path.name, start_idx=warp_idx)

        # Reconocimiento y dibujo
        for warp_info in warps:
            approx = warp_info["approx"]
            warped_img = warp_info["warped"]

            rank, suit, r_score, s_score = recognize_card_from_warp(warped_img, templates_rank, templates_suit)
            label = f"{rank}_{suit}" if rank != "UNKNOWN" and suit != "UNKNOWN" else "UNKNOWN"
            draw_bbox_with_label(img_bgr, approx, label)

        # Guardar imagen resultante
        save_image(results_path / f"{img_path.stem}_det.jpg", img_bgr)
        print(f"  Guardado: {results_path / f'{img_path.stem}_det.jpg'}")
        input("Presiona ENTER para continuar con la siguiente imagen...")

    print("Procesado completado.")

if __name__ == "__main__":
    main()
