import inkml
import pickle
import cv2
import token_parser
import inkml_graphics
import generator
import random
import numpy as np
import file_utils as utils
import os

random.seed(123)
np.random.seed(123)

#
# def _filter(trace):
#     prev_x = -1
#     prev_y = -1
#     prev_eq_x = 0
#     prev_eq_y = 0
#     for point in trace:
#         x = point[0]
#         y = point[1]
#
#         if prev_x == x:
#             prev_eq_x += 1
#         else:
#             prev_eq_x = 0
#
#         if prev_y == y:
#             prev_eq_y += 1
#         else:
#             prev_eq_y = 0
#
#         prev_x = x
#         prev_y = y
#         if prev_eq_x == 3 or prev_eq_y == 3:
#             return True
#     return False

overfit_images, truth, _ = zip(*utils.read_pkl(os.path.join('/Users/balazs/new_data', "overfit.pkl")))

gen = generator.simple_number_operation_generator()
conf = generator.Config()
#parser = token_parser.Parser(inkml_graphics.create_graphics_factory('/Users/balazs/export/tokengroup_test2011.pkl'))
parser = token_parser.Parser(inkml_graphics.create_graphics_factory('/Users/balazs/export/tokengroup.pkl'))

while True:
    tokens = []
    gen.generate(tokens, conf)
    print(tokens)
    image = parser.parse(tokens)

    image = 255 - image

    imid = np.random.choice(len(overfit_images))
    overfit_im = overfit_images[imid]
    cv2.imshow("Image", image)
    cv2.imshow("Overfit", overfit_im)
    cv2.waitKey(0)
# for key in token_traces.keys():
#     traces = token_traces[key]
#     new_traces = []
#
#     for index, trace_group in enumerate(traces):
#         add = True
#         for trace in trace_group:
#             if _filter(trace):
#                 add = False
#                 break
#         if add:
#             new_traces.append(trace_group)
#     token_traces[key] = new_traces


# for key in token_traces.keys():
#     traces = token_traces[key]
if True:
    key = '\\sqrt'
    traces = token_traces[key]
    print(key)
    for i in range(min(len(traces), 10)):
        print(traces[i][0])
        image = graphics.create_token_image(token_traces[key][i], expected_width=400, expected_height=100)
        cv2.imshow("Image", image)
        cv2.waitKey(0)