from graphics.utils import *
import cv2


class Graphics:

    def __init__(self):
        self.image = None
        self.current_point = None
        self.max = (0, 0)
        self.min = (0, 0)


    def start_image(self, inkml):
        self.current_point = None
        for symbol in inkml.symbols:
            for point in symbol:
                self.max = (max(self.max[0], point[0]), max(self.max[1], point[1]))
                self.min = (min(self.min[0], point[0]), min(self.min[1], point[1]))
        self.image = new_image(self.max[0], self.max[1])


    def new_symbol(self):
        self._assert_start_image_called()
        self.current_point = None


    def point(self, point):
        self._assert_start_image_called()
        if self.current_point is not None:
            cv2.line(self.image, self.current_point, point, (0, 0, 0), 20)
            # line between point and first_point
        self.current_point = point

    def end_image(self):
        self._assert_start_image_called()
        image = self.image
        self.image = None
        return image

    def _assert_start_image_called(self):
        assert self.image is not None, "did you forget to call start_image(inkml)?"
