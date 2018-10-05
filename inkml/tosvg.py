import svgwrite
import numpy as np
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
    scale = 2.0 / mean_distance

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

