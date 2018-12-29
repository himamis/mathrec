from graphics.utils import *
import cv2
import math

info = np.finfo(np.float32)

def normalize_points(inkml):
    min_x = info.max
    min_y = info.max
    max_x = info.min
    max_y = info.min

    mean_distance = 0.0
    total_points = 0

    for k in range(len(inkml)):
        symbol = inkml[k]
        trace_length = 0
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
                trace_length += distance
                total_points += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    size_x = max_x - min_x
    size_y = max_y - min_y

    mean_distance /= total_points
    scale = 20.0 / mean_distance

    trans_x = -min_x
    trans_y = -min_y

    width = size_x * scale + 20
    height = size_y * scale + 20


    for k in range(len(inkml)):
        symbol = inkml[k]

        for j in range(len(symbol)):
            x = symbol[j][0]
            y = symbol[j][1]

            x = (x + trans_x) * scale + 10
            y = (y + trans_y) * scale + 10

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

    def create_token_image(self, traces, expected_width=None, expected_height=None, padding=10):
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

        width = None
        height = None
        scale_x = None
        scale_y = None
        if expected_width is not None:
            scale_x = (expected_width - padding * 2) / (max_x - min_x)
            width = expected_width

        if expected_height is not None:
            scale_y = (expected_height - padding * 2) / (max_y - min_y)
            height = expected_height

        scale_x = scale_y if scale_x is None else scale_x
        scale_y = scale_x if scale_y is None else scale_y

        trans_x = min_x
        trans_y = min_y

        if width is None:
            width = round((max_x - trans_x) * scale_x) + 2 * padding
        if height is None:
            height = round((max_y - trans_y) * scale_y) + 2 * padding

        # Normalize points
        for trace_index, trace in enumerate(traces):
            new_trace = []
            prev_point = None
            for point in trace:
                x = (point[0] - trans_x) * scale_x + padding
                y = (point[1] - trans_y) * scale_y + padding

                new_point = np.array((x, y), dtype=np.float32)
                if np.any(new_point != prev_point):
                    new_trace.append(new_point)
                prev_point = new_point
            traces[trace_index] = new_trace

        image = new_image(width, height)

        pts = [np.rint(line) for line in traces]
        pts = [np.asarray(line, dtype=np.int32) for line in pts]
        pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in pts]
        cv2.polylines(image, pts, False, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        return image

