import tkinter as tk
from gui import simple_dialog
from transformer import trainer
import threading
import queue

q = queue.Queue()

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

    def __init__(self, model, res_var, transformer_input):
        self.model = model
        self.res_var = res_var
        self.transformer_input = transformer_input

    def start_evaluating(self):
        while True:
            inp = self.transformer_input.to_input()
            if len(inp) > 0:
                self.run_model(inp)

    def run_model(self, input):
        print("Running model")
        result = trainer.transform(self.model, input)
        q.put(result)




transformer_input = TransformerInput()

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
        transformer_input.add(self.rect, value, self.bbox)
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
        transformer_input.change_box(self.handle, c)

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
        transformer_input.remove(self.handle)
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
    transformer_model = trainer.prepare_transform()
    eval_c = EvaluationController(transformer_model, res_var, transformer_input)
    label_var.set("Ready!")
    threading.Thread(target=eval_c.start_evaluating).run()

def clear_canvas():
    print("Clear canvas")
    canvas.delete(tk.ALL)
    transformer_input.clear()


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

transformer_model = None


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
