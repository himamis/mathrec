import numpy as np
import cv2


def image_from_bytes(bytes):
    buf = np.frombuffer(bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.asarray(img)