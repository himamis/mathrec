from graphics.utils import *
import cv2
import math
import copy

info = np.finfo(np.float32)


class CoordTransform(object):

    def __init__(self):
        self.translate = 0
        self.scale = 0

    def transform(self, coord):
        raise Exception("Not implemented")


class TranslateScaleTransformer(CoordTransform):

    def __init__(self, translate, scale):
        super().__init__()
        self.translate = translate
        self.scale = scale

    def transform(self, coord):
        return (coord - self.translate) * self.scale


class ReplaceTransform(CoordTransform):

    def __init__(self, length):
        super().__init__()
        self.length = length

    def transform(self, coord):
        return int(round(self.length / 2))


def normalize_points(inkml, unit=False):
    min_x = info.max
    min_y = info.max
    max_x = info.min
    max_y = info.min

    mean_distance = 0.0
    total_points = 0

    for i in range(len(inkml)):
        tracegroup = inkml[i]
        for k in range(len(tracegroup)):
            symbol = tracegroup[k]
            for j in range(len(symbol)):
                x = symbol[j][0]
                y = symbol[j][1]
                if j != 0:
                    if x == symbol[j-1][0] and y == symbol[j-1][1]:
                        continue
                    delta_x = x - symbol[len(symbol) - 1][0]
                    delta_y = y - symbol[len(symbol) - 1][1]
                    distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                    mean_distance += distance
                    total_points += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    size_x = max_x - min_x
    size_y = max_y - min_y

    trans_x = -min_x
    trans_y = -min_y

    if unit:
        scale_x = 1 / size_x
        scale_y = 1 / size_y

        width = 1
        height = 1

        padding_x = 0
        padding_y = 0
    else:
        mean_distance /= total_points
        scale_x = 20.0 / mean_distance
        scale_y = scale_x

        padding_x = 20
        padding_y = 20

        width = size_x * scale_x + padding_x
        height = size_y * scale_y + padding_y

    for i in range(len(inkml)):
        tracegroup = inkml[i]
        for k in range(len(tracegroup)):
            symbol = tracegroup[k]

            for j in range(len(symbol)):
                x = symbol[j][0]
                y = symbol[j][1]

                x = (x + trans_x) * scale_x + padding_x / 2
                y = (y + trans_y) * scale_y + padding_y / 2

                symbol[j] = np.array((x, y), dtype=np.float32)

    return (width, height)

class Graphics:

    def create_image(self, inkml):
        (width, height) = normalize_points(inkml)

        image = new_image(int(width + 1), int(height + 1))

        #pts = [line - point_min for line in inkml.symbols]
        pts = [np.rint(line) for line in inkml]
        pts = [np.asarray(line, dtype=np.int32) for line in pts]
        pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in pts]
        cv2.polylines(image, pts, False, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return image

    def create_token_image(self, traces, expected_width=None, expected_height=None, padding=6):
        if expected_height is None and expected_width is None:
            print("Must provide height of width at least")
            exit(0)

        # Find min max and scale
        min_x = info.max
        min_y = info.max
        max_x = info.min
        max_y = info.min
        for trace in traces:
            for point in trace:
                min_x = min(min_x, point[0])
                min_y = min(min_y, point[1])
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])

        if max_x == min_x and max_y == min_y:
            return self._create_point(expected_width, expected_height, padding)

        width = None
        height = None

        scale_x = 0
        scale_y = 0
        transform_x = None
        transform_y = None

        if expected_width is not None:
            if max_x == min_x:
                transform_x = ReplaceTransform(expected_width)
            else:
                scale_x = (expected_width - padding * 2) / (max_x - min_x)
                transform_x = TranslateScaleTransformer(min_x, scale_x)

            width = expected_width

        if expected_height is not None:
            if max_y == min_y:
                transform_y = ReplaceTransform(expected_height)
            else:
                scale_y = (expected_height - padding * 2) / (max_y - min_y)
                transform_y = TranslateScaleTransformer(min_y, scale_y)
            height = expected_height

        if transform_x is None:
            transform_x = copy.copy(transform_y)
            transform_x.translate = min_x
            scale_x = scale_y

        if transform_y is None:
            transform_y = copy.copy(transform_x)
            transform_y.translate = min_y
            scale_y = scale_x

        if width is None:
            width = int(round((max_x - min_x) * scale_x + 2 * padding))
        if height is None:
            height = int(round((max_y - min_y) * scale_y + 2 * padding))

        # Normalize points
        for trace_index, trace in enumerate(traces):
            new_trace = []
            prev_point = None
            for point in trace:
                x = transform_x.transform(point[0]) + padding
                y = transform_y.transform(point[1]) + padding
                #x = (point[0] - trans_x) * scale_x + padding
                #y = (point[1] - trans_y) * scale_y + padding

                new_point = np.array((x, y), dtype=np.float32)
                if np.any(new_point != prev_point):
                    new_trace.append(new_point)
                prev_point = new_point
            traces[trace_index] = new_trace

        image = new_image(width, height)

        # Draw points
        for trace in traces:
            if len(trace) == 1:
                # It's a point
                point = trace[0]
                self._draw_point(image, (point[0], point[1]), np.random.randint(2, 3))

        pts = [np.rint(line) for line in traces]
        pts = [np.asarray(line, dtype=np.int32) for line in pts]
        pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in pts]
        cv2.polylines(image, pts, False, (0, 0, 0), 2, lineType=cv2.LINE_4)

        return image

    def _create_point(self, expected_width=None, expected_height=None, padding=10):
        if expected_width is None:
            expected_width = 10
        if expected_height is None:
            expected_height = 10
        new_im = new_image(expected_width, expected_height)
        width, height = int(round(expected_width / 2)), int(round(expected_height / 2))
        self._draw_point(new_im, (width, height), min(width - padding, height - padding))

        return new_im

    def _draw_point(self, image, location, radius):
        cv2.circle(image, location, radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
