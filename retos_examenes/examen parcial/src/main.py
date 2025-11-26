import cv2
from pathlib import Path
from src.segmentacion import detect_cards_in_image
from src.reconocimiento import load_templates, recognize_card_from_warp
from src.utils import draw_bbox_with_label, save_image, ensure_dir
from src.config import RAW_DIR, RESULT_IMAGES, PROCESSED_DIR

def main():
    raw_path = Path(RAW_DIR)
    results_path = Path(RESULT_IMAGES)
    processed_path = Path(PROCESSED_DIR)
    ensure_dir(results_path)
    ensure_dir(processed_path)

    print(f"[DEBUG] Buscando imágenes en {raw_path}")
    image_files = sorted([f for f in raw_path.glob("*.*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    print(f"[DEBUG] {len(image_files)} imágenes encontradas.")

    if len(image_files) == 0:
        print("No se encontraron imágenes en RAW_DIR. Terminado.")
        return

    templates = load_templates()
    if not templates:
        print("ERROR: No se encontraron templates")
        return
    print(f"[DEBUG] {sum(len(v) for v in templates.values())} templates cargados por palo.")

    warp_idx = 0
    for img_path in image_files:
        print(f"[DEBUG] Procesando {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"ERROR: No se pudo leer la imagen {img_path.name}")
            continue

        warps, _, warp_idx = detect_cards_in_image(img_bgr, img_name=img_path.stem, start_idx=warp_idx)
        print(f"[DEBUG] {len(warps)} cartas detectadas en la imagen {img_path.name}")

        for w in warps:
            warp_img = w["warped"]

            # Guardar cada warp en processed (RGB)
            warp_file = processed_path / w["warp_path"]
            cv2.imwrite(str(warp_file), warp_img)
            print(f"[DEBUG] Warp guardado en {warp_file}")

            numero, palo, score = recognize_card_from_warp(warp_img, templates)
            label = f"{numero} de {palo}" if numero != "UNKNOWN" else "UNKNOWN"
            draw_bbox_with_label(img_bgr, w["approx"], label)
            print(f"[DEBUG] Carta reconocida: {label}, score={score:.2f}")

        save_image(results_path / f"{img_path.stem}_det.jpg", img_bgr)
        print(f"[DEBUG] Imagen resultante guardada en {results_path / f'{img_path.stem}_det.jpg'}")

    print("Procesado completado.")

if __name__ == "__main__":
    main()
