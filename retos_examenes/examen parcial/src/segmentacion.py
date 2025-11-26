import cv2
import numpy as np
from pathlib import Path

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
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp = cv2.warpPerspective(img_bgr, M, (width, height))

        warp_name = f"{img_name}_warp_{warp_idx:03d}.png"
        warps.append({"warped": warp, "warp_path": warp_name, "approx": box})

        # Guardar en processed en RGB
        if processed_path is not None:
            processed_path.mkdir(parents=True, exist_ok=True)
            warp_rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(processed_path / warp_name), warp_rgb)

        warp_idx += 1

    return warps, thresh, warp_idx
