import numpy as np
import cv2
from graphics import *


kernel = np.ones((3, 3), np.uint8)


class Postprocessor:

    def postprocess(self, image):
        #dilation = cv2.erode(image, kernel, iterations=1)
        #image = cv2.blur(image, (3,3))
        _, image = cv2.threshold(image, 230, 255, cv2.THRESH_TOZERO)
        pads = np.random.randint(10, 50, 4)
        image = pad_image(image, pads[0], pads[1], pads[2], pads[3])
        #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
        #image = cv2.GaussianBlur(image, (1, 1), 0)
        #image = cv2.bilateralFilter(image, 9, 75, 75)
        return image
