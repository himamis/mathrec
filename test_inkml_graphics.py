import inkml
import pickle
import cv2

token_traces = pickle.load(open('/Users/balazs/export/tokengroup_test2011.pkl', 'rb'))
graphics = inkml.Graphics()

def _filter(trace):
    prev_x = -1
    prev_y = -1
    prev_eq_x = 0
    prev_eq_y = 0
    for point in trace:
        x = point[0]
        y = point[1]

        if prev_x == x:
            prev_eq_x += 1
        else:
            prev_eq_x = 0

        if prev_y == y:
            prev_eq_y += 1
        else:
            prev_eq_y = 0

        prev_x = x
        prev_y = y
        if prev_eq_x == 3 or prev_eq_y == 3:
            return True
    return False


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