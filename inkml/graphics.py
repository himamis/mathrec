from graphics.utils import *
import cv2
import math

def normalize_points(inkml):
    info = np.finfo(np.float32)

    min_x = info.max
    min_y = info.max
    max_x = info.min
    max_y = info.min

    mean_distance = 0.0
    total_points = 0

    for k in range(len(inkml.symbols)):
        symbol = inkml.symbols[k]
        trace_length = 0
        for j in range(len(symbol)):
            x, y = symbol[j]
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


    for k in range(len(inkml.symbols)):
        symbol = inkml.symbols[k]

        for j in range(len(symbol)):
            x, y = symbol[j]

            x = (x + trans_x) * scale + 10
            y = (y + trans_y) * scale + 10

            symbol[j] = np.array((x, y), dtype=np.float32)

    return (width, height)

class Graphics:

    def create_image(self, inkml):
        (width, height) = normalize_points(inkml)

        image = new_image(int(width + 1), int(height + 1))

        #pts = [line - point_min for line in inkml.symbols]
        pts = [np.rint(line) for line in inkml.symbols]
        pts = [np.asarray(line, dtype=np.int32) for line in pts]
        pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in pts]
        cv2.polylines(image, pts, False, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        return image


    def create_svg(self, inkml, filename):
        #svg = svgwrite.Drawing(filename, debug=False)
        #for symbol in inkml.symbols:
        #    svg.add(svg.polyline(symbol))
        #svg.save()
        svg = self.js()