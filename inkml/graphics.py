from graphics.utils import *
import cv2


class Graphics:

    def create_image(self, inkml):
        info = np.iinfo(np.int32)
        zero_point = np.zeros(2, dtype=np.int32)
        point_min = np.array((info.max, info.max), dtype=np.int32)
        point_max = zero_point
        for symbol in inkml.symbols:
            for point in symbol:
                point_max = np.maximum(point_max, point)
                point_min = np.minimum(point_min, point)
        margin = np.array((20, 20), dtype=np.int32)
        point_max = point_max + margin
        point_min = point_min - margin

        point_min = np.maximum(zero_point, point_min)

        size = point_max - point_min

        image = new_image(size[0], size[1])

        pts = [line - point_min for line in inkml.symbols]
        pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in pts]
        cv2.polylines(image, pts, False, (0, 0, 0), 3, lineType=cv2.LINE_AA)

        return image
