import tkinter as tk
from gui import simple_dialog
import threading
import queue
import h2o
import numpy as np

info = np.finfo(np.float32)


def normalize(inputs, padding=0.0001):
    min_x = info.max
    min_y = info.max
    max_x = info.min
    max_y = info.min

    for token, bounding_box in inputs:
        min_x = min(min_x, bounding_box[0])
        min_y = min(min_y, bounding_box[1])
        max_x = max(max_x, bounding_box[2])
        max_y = max(max_y, bounding_box[3])

    trans_x = min_x - padding / 2.0
    trans_y = min_y - padding / 2.0
    scale_x = (1 - padding) / (max_x - min_x)
    scale_y = (1 - padding) / (max_y - min_y)
    result = []
    for token, bounding_box in inputs:
        box = ((bounding_box[0] - trans_x) * scale_x,
               (bounding_box[1] - trans_y) * scale_y,
               (bounding_box[2] - trans_x) * scale_x,
               (bounding_box[3] - trans_y) * scale_y)
        result.append((token, box))

    return result


def normalize_boxes(boxes, to_size=0.01):
    if len(boxes) == 1:
        return boxes
    min_x_dist = info.max
    min_y_dist = info.max
    for i in range(len(boxes) - 1):
        for j in range(i + 1, len(boxes)):
            token1, box1 = boxes[i]
            token2, box2 = boxes[j]
            dists_x = (box1[0] - box2[0], box1[0] - box2[2], box1[2] - box2[0], box1[2] - box2[2])
            dists_y = (box1[1] - box2[1], box1[1] - box2[3], box1[3] - box2[1], box1[3] - box2[3])
            abs_dists_x = [abs(dist) for dist in dists_x if dist != 0]
            abs_dists_y = [abs(dist) for dist in dists_y if dist != 0]
            if len(abs_dists_y) == 0:
                abs_dists_y = [to_size]
            if len(abs_dists_x) == 0:
                abs_dists_x = [to_size]
            smallest_x = min(abs_dists_x)
            smallest_y = min(abs_dists_y)

            min_x_dist = min(min_x_dist, smallest_x)
            min_y_dist = min(min_y_dist, smallest_y)
    scale_x = to_size / min_x_dist
    scale_y = to_size / min_y_dist

    for index, (token, box) in enumerate(boxes):
        boxes[index] = (token, (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y))
    return boxes

# Main algorithm for sequence data
def key(inp):
    token, bounding = inp
    if token == '(' or token == '\\[' or token == '\\{':
        return bounding[0]
    elif token == ')' or token == '\\]' or token == '\\}':
        return bounding[2]
    else:
        return (bounding[0] + bounding[2]) / 2

def standardize_data(bounding_boxes):
    bounding_boxes = normalize(bounding_boxes)
    return sorted(bounding_boxes, key=key)

def create_features(boxes):
    boxes = standardize_data(boxes)
    data = []
    for i in range(len(boxes) - 1):
        for j in range(i + 1, len(boxes)):
            ftok, b1 = boxes[i]
            ttok, b2 = boxes[j]
            data += [(b1[0] - b2[0], b1[0] - b2[2], b1[2] - b2[0], b1[2] - b2[2],
                     b1[1] - b2[1], b1[1] - b2[3], b1[3] - b2[1], b1[3] - b2[3],
                      b1[0], b1[1], b1[2], b1[3], b2[0], b2[1], b2[2], b2[3],
                     ftok, ttok)]
    return h2o.H2OFrame.from_python(data, column_types=['numeric', 'numeric', 'numeric', 'numeric',
                                                        'numeric', 'numeric', 'numeric', 'numeric',
                                                        'numeric', 'numeric', 'numeric', 'numeric',
                                                        'numeric', 'numeric', 'numeric', 'numeric',
                                                        'factor', 'factor'],
                                   column_names=['minx-minx','minx-maxx','maxx-minx', 'maxx-maxx',
              'miny-miny', 'miny-maxy', 'maxy-miny', 'maxy-maxy',
                                                 'rminx', 'rminy', 'rmaxx', 'rmaxy',
                                                 'tminx', 'tminy', 'tmaxx', 'tmaxy',
                                                 'relative_to', 'this'])



#h2o.connect()
h2o.init()
q = queue.Queue()
input_action = queue.Queue()

class TransformerInput:

    def __init__(self):
        self.input = {}
        self.listeners = []

    def remove(self, index):
        del self.input[index]
        self._notify_listeners()

    def add(self, index, token, bounding_box):
        self.input[index] = (token, bounding_box)
        self._notify_listeners()

    def change_box(self, index, bounding_box):
        self.input[index] = (self.input[index][0], bounding_box)
        self._notify_listeners()

    def clear(self):
        self.input = {}
        self._notify_listeners()

    def to_input(self):
        result = []
        for token, bounding_boxes in self.input.values():
            result.append((token, bounding_boxes))
        return result

    def add_listener(self, listener):
        self.listeners.append(listener)

    def _notify_listeners(self):
        for listener in self.listeners:
            listener.input_changed(self)


class EvaluationController:

    def __init__(self, model, res_var):
        self.model = model
        self.res_var = res_var
        self.transformer_input = TransformerInput()

    def start_evaluating(self):
        while True:
            a = input_action.get()
            self.do_action(a)
            with input_action.mutex:
                input_action.queue.clear()
            inp = self.transformer_input.to_input()
            if len(inp) > 0:
                self.run_model(inp)

    def do_action(self, action):
        action, params = action
        if action == "remove":
            self.transformer_input.remove(params)
        elif action == "add":
            (rect, value, bbox) = params
            self.transformer_input.add(rect, value, bbox)
        elif action == "change_box":
            (handle, c) = params
            self.transformer_input.change_box(handle, c)
        elif action == "clear":
            self.transformer_input.clear()
        else:
            print("UNKNOWN ACTION")



    def run_model(self, input):
        print("Running model")
        # print(input)
        features = create_features(input)
        #print(features)
        result = self.model.predict(features)
        #print(result)
        if len(result) != 0:
            print(result.concat(features[-2:]))
        #result = "csa"
        #q.put(result)


class InputDialog(simple_dialog.Dialog):

    def __init__(self, parent, callback):
        self.callback = callback
        super().__init__(parent)

    def body(self, master):
        tk.Label(master, text="First:").grid(row=0)

        self.e1 = tk.Entry(master)

        self.e1.grid(row=0, column=1)
        return self.e1  # initial focus

    def apply(self):
        first = self.e1.get()
        self.callback(first)


class MouseController(object):

    def __init__(self, canvas):
        self.canvas = canvas

    def mouse_down(self, event):
        pass

    def mouse_released(self, event):
        pass

    def mouse_moved(self, event):
        pass

    def mouse_cancelled(self, event):
        pass


class CreatorController(MouseController):

    def __init__(self, canvas):
        self.x = 0
        self.y = 0
        self.lastx = 0
        self.lasty = 0
        self.rect = 0
        self.bbox = ()
        self.done = False
        super().__init__(canvas)

    def mouse_down(self, event):
        self.canvas.config(cursor='cross')
        self.x = event.x
        self.y = event.y
        self.rect = self.canvas.create_rectangle(self.x, self.y, self.x, self.y)
        self.canvas.itemconfig(self.rect, tags=("index_" + str(self.rect), "token", self.rect))

    def mouse_moved(self, event):
        self.lastx = event.x
        self.lasty = event.y
        self.canvas.coords(self.rect, self.x, self.y, self.lastx, self.lasty)

    def mouse_released(self, event):
        self.canvas.config(cursor='')
        self.bbox = (min(self.x, self.lastx), min(self.y, self.lasty),
                     max(self.x, self.lastx), max(self.y, self.lasty))

        InputDialog(self.canvas, self._ok_callback)
        if not self.done:
            self.canvas.delete(self.rect)
        else:
            self.done = False

    def mouse_cancelled(self, event):
        self.canvas.delete(self.rect)
        self.done = False
        self.canvas.config(cursor='')

    def _ok_callback(self, value):
        input_action.put(("add", (self.rect, value, self.bbox)))
        # transformer_input.add(self.rect, value, self.bbox)
        tag = "index_" + str(self.rect)
        self.canvas.create_text(self.bbox[0], self.bbox[1]-10, text=value, tags=(tag, "text", self.rect))
        self.canvas.create_line(self.bbox[2] - 10, self.bbox[1] - 10, self.bbox[2], self.bbox[1], tags=(tag, "delete", self.rect))
        self.canvas.create_line(self.bbox[2] - 10, self.bbox[1], self.bbox[2], self.bbox[1] - 10, tags=(tag, "delete", self.rect))
        self.done = True


class MovingController(MouseController):

    def __init__(self, canvas, handle):
        self.handle = handle
        self.x = 0
        self.y = 0
        super().__init__(canvas)

    def mouse_down(self, event):
        self.canvas.config(cursor='sizing')
        self.x = event.x
        self.y = event.y

    def mouse_moved(self, event):
        delta_x = event.x - self.x
        delta_y = event.y - self.y

        indices = self.canvas.find_withtag("index_" + str(self.handle))
        for index in indices:
            self._move_item(index, delta_x, delta_y)

        c = canvas.coords(self.handle)
        input_action.put(("change_box", (self.handle, c)))
        #transformer_input.change_box(self.handle, c)

        self.x = event.x
        self.y = event.y

    def _move_item(self, index, delta_x, delta_y):
        c = canvas.coords(index)
        if len(c) == 4:
            self.canvas.coords(index, c[0] + delta_x, c[1] + delta_y, c[2] + delta_x, c[3] + delta_y)
        elif len(c) == 2:
            self.canvas.coords(index, c[0] + delta_x, c[1] + delta_y)
        else:
            print("Coords len " + str(len(c)))

    def mouse_released(self, event):
        self.canvas.config(cursor='')
        pass


class DeleteController(MouseController):

    def __init__(self, canvas, handle):
        self.canvas = canvas
        self.handle = handle
        self.x = 0
        self.y = 0

    def mouse_down(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.config(cursor='cross')

    def mouse_moved(self, event):
        if self._test(event):
            self.canvas.config(cursor='cross')
        else:
            self.canvas.config(cursor='')
        pass

    def mouse_released(self, event):
        if self._test(event):
            self._delete()
        self.canvas.config(cursor='')

    def _test(self, event):
        return abs(event.x - self.x) < 10 and abs(event.y - self.y) < 10

    def mouse_cancelled(self, event):
        self.canvas.config(cursor='')

    def _delete(self):
        input_action.put(("remove", self.handle))
        # transformer_input.remove(self.handle)
        indices = self.canvas.find_withtag("index_" + str(self.handle))
        for index in indices:
            self.canvas.delete(index)


class RootMouseController(MouseController):

    def __init__(self, canvas):
        self.creator_controller = CreatorController(canvas)
        self.active_controller = None
        self.mouse_down_called = False
        self.mouse_cancelled_called = False
        super().__init__(canvas)

    def mouse_down(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        found = self.canvas.find_overlapping(x-20, y-20, x+5, y+5)
        if len(found) > 0:
            index = found[0]
            tags = self.canvas.gettags(index)
            if tags[1] == "delete":
                self.active_controller = DeleteController(canvas, int(tags[2]))
            elif tags[1] == "token":
                self.active_controller = MovingController(canvas, int(tags[2]))
            else:
                self.active_controller = self.creator_controller
        else:
            self.active_controller = self.creator_controller

        self.active_controller.mouse_down(event)
        self.mouse_down_called = True

    def mouse_moved(self, event):
        self.active_controller.mouse_moved(event)

    def mouse_released(self, event):
        if not self.mouse_cancelled_called:
            self.active_controller.mouse_released(event)
        self.mouse_down_called = False
        self.mouse_cancelled_called = False

    def key_pressed(self, event):
        if event.keysym == 'Escape' and self.mouse_down_called:
            self.active_controller.mouse_cancelled(event)
            self.mouse_cancelled_called = True
        self.mouse_down_called = False


def create_canvas(top):
    canvas = tk.Canvas(top)
    controller = RootMouseController(canvas)
    canvas.bind("<Button-1>", controller.mouse_down)
    canvas.bind("<ButtonRelease-1>", controller.mouse_released)
    canvas.bind("<B1-Motion>", controller.mouse_moved)
    canvas.bind("<Key>", controller.key_pressed)
    canvas.pack(fill=tk.BOTH, expand=True)
    return canvas


def create_text(top):
    text = tk.Label(top, text="csa")
    text.pack(side=tk.BOTTOM, fill=tk.X, expand=False)


def load_transformer_background():
    t = threading.Thread(target=load_transformer)
    t.start()

def load_transformer():
    label_var.set("Loading...")
    transformer_model = h2o.load_model(
        "/Users/balazs/Documents/notebooks/transformer-naive/DRF_model_python_1556873193433_19"
        #"/Users/balazs/Documents/notebooks/transformer-naive/DRF_model_python_1556873193433_3"
    )
    eval_c = EvaluationController(transformer_model, res_var,)
    label_var.set("Ready!")
    threading.Thread(target=eval_c.start_evaluating).run()

def clear_canvas():
    print("Clear canvas")
    canvas.delete(tk.ALL)
    input_action.put(("clear", None))
    # transformer_input.clear()


def create_buttons(top):
    frame = tk.Frame(top)
    frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True)

    state_label = tk.Label(frame, textvariable=label_var)
    state_label.grid(row=0, column=2, padx=20)

    button = tk.Button(frame, text="Clear canvas", command=clear_canvas)
    button.grid(row=0, column=0)

    button = tk.Label(frame, textvariable=res_var)
    button.grid(row=0, column=1)

    return state_label

def poll():
    try:
        res = q.get_nowait()
        res_var.set(res)
    except queue.Empty:
        pass
    top.after(100, poll)

model = None


top = tk.Tk()
label_var = tk.StringVar(top)
res_var = tk.StringVar(top)
top.wm_minsize(600, 400)
canvas = create_canvas(top)
create_text(top)
create_buttons(top)
canvas.focus_set()

load_transformer_background()
poll()
top.mainloop()
