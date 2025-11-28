import cv2
import numpy as np
from pathlib import Path
from src.config import CORNER_CROP

CARD_WIDTH = 200
CARD_HEIGHT = 300

def detect_cards_in_image(img_bgr, img_name="img", start_idx=0, processed_path: Path = None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    warps = []
    warp_idx = start_idx

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)

        w, h = rect[1]
        angle = rect[2]
        if w > h:
            w, h = h, w
            angle += 90

        dst_pts = np.array([[0, 0], [CARD_WIDTH-1, 0], [CARD_WIDTH-1, CARD_HEIGHT-1], [0, CARD_HEIGHT-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(box, dst_pts)
        warp = cv2.warpPerspective(img_bgr, M, (CARD_WIDTH, CARD_HEIGHT))

        # Extraer corner según CORNER_CROP
        y1, y2, x1, x2 = CORNER_CROP
        corner = warp[y1:y2, x1:x2]

        warp_name = f"{img_name}_warp_{warp_idx:03d}.png"
        warps.append({"warped": warp, "corner": corner, "warp_path": warp_name, "approx": box.astype(np.int32)})

        if processed_path is not None:
            processed_path.mkdir(parents=True, exist_ok=True)
            warp_rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(processed_path / warp_name), warp_rgb)

        warp_idx += 1

    return warps, thresh, warp_idx
