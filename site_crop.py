import cv2
import numpy as np

AREA_MM = [(60, -170), (60, 170), (260, 170), (260, -170)]
MARGIN_PX = 40


def crop_box(H, width=1920, height=1080):
    corners = cv2.perspectiveTransform(np.float32(AREA_MM)[None], np.linalg.inv(H))[0]
    x0, y0 = np.maximum(corners.min(axis=0) - MARGIN_PX, 0).astype(int)
    x1, y1 = np.minimum(corners.max(axis=0) + MARGIN_PX, (width, height)).astype(int)
    return int(x0) // 2 * 2, int(y0) // 2 * 2, int(x1) // 2 * 2, int(y1) // 2 * 2


def crop(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]
