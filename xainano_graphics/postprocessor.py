import numpy as np
import cv2


kernel = np.ones((3, 3), np.uint8)


class Postprocessor:

    def postprocess(self, image):

        dilation = cv2.erode(image, kernel, iterations=1)
        return dilation
