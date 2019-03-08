import tensorflow as tf
from database_create.methods import *
from utilities import progress_bar
import pickle
from inkml import graphics as g
import cv2
import numpy as np
import graphics.utils as u
import os
import graphics.augment as a

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_label_map(vocabulary):
    label_map = ""
    for name, id in vocabulary.items():
        if name == "\"":
            name = "\\\""
        label_map += "item {\n  name: \"" + name + "\"\n  id: " + str(id) + "\n}\n"
    return label_map


def write_label_map(encoding):
    label_map = create_label_map(encoding)
    # Write label map
    with open(os.path.join(FLAGS.output_path, "dataset_label_map.pbtxt"), 'w') as file:
        file.write(label_map)


def create_tf_example(example):
    image = example['image']
    cv2.imshow("image", image)
    cv2.waitKey(0)

    height = u.h(image)  # Image height
    width = u.w(image)  # Image width
    filename = b""  # Filename of the image. Empty if image is not from file
    encoded_image_data = example['encoded_image']  # Encoded image bytes
    image_format = example['format']  # b'jpeg' or b'png'

    bounding_boxes = example['bounding_boxes']
    xmins = [box[0] for box in bounding_boxes]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [box[2] for box in bounding_boxes]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [box[1] for box in bounding_boxes]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [box[3] for box in bounding_boxes]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = example['truths']  # List of string class name of bounding box (1 per box)
    classes = example['truth_ids']  # List of integer class id of bounding box (1 per box)

    # tf_example = tf.train.Example(features=tf.train.Features(feature={
    #     'image/height': dataset_util.int64_feature(height),
    #     'image/width': dataset_util.int64_feature(width),
    #     'image/filename': dataset_util.bytes_feature(filename),
    #     'image/source_id': dataset_util.bytes_feature(filename),
    #     'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    #     'image/format': dataset_util.bytes_feature(image_format),
    #     'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    #     'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    #     'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    #     'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    #     'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    #     'image/object/class/label': dataset_util.int64_list_feature(classes),
    # }))
    return None


def calculate_bounding_boxes(inkml):
    bounding_boxes = []
    info = np.iinfo(np.int32)

    for tracegroup in inkml:
        minx = info.max
        miny = info.max
        maxx = info.min
        maxy = info.min

        for trace in tracegroup:
            for point in trace:
                assert len(point) == 1
                point = point[0]
                minx = min(minx, point[0])
                miny = min(miny, point[1])
                maxx = max(maxx, point[0])
                maxy = max(maxy, point[1])

        bounding_boxes.append((minx, miny, maxx, maxy))

    return bounding_boxes


def normalize_bounding_boxes(bounding_boxes, width, height):
    new_boxes = []
    scalex = float(width)
    scaley = float(height)

    # Normalize bounding boxes
    for index, box in enumerate(bounding_boxes):
        box = (
            float(box[0]) / scalex,
            float(box[1]) / scaley,
            float(box[2]) / scalex,
            float(box[3]) / scaley
        )

        assert box[0] >= 0 and box[1] >= 0 and box[2] <= 1 and box[3] <= 1
        new_boxes.append(box)

    return new_boxes


def draw_rectangles(image, boxes):
    #for box in boxes:
    #    cv2.rectangle(image, (
    #        box[0] * float(u.w(image))), box[1]), (box[2], box[3]), (0, 255, 0), 1)
    pass


def convert_truth(truth):
    if truth == '<':
        return '\\lt'
    elif truth == '>':
        return '\\gt'
    return truth


def get_query(train=True):
    return "select formula.id, formula.writerid, formula.formula FROM public.database, public.writer, " \
           "public.formula WHERE public.formula.writerid = public.writer.id " \
           "AND public.writer.databaseid = public.database.id " \
           "AND public.database.name" + (" NOT" if train else "") + " LIKE 'CROHME2016_data/Test2016_INKML_GT';"


def translate_to(points, padding=10):
    min_x = 99999
    min_y = 99999
    for symbol in points:
        for point in symbol:
            x = point[0][0]
            y = point[0][1]

            min_x = min(x, min_x)
            min_y = min(y, min_y)

    for symbol in points:
        for index, point in enumerate(symbol):
            x = point[0][0]
            y = point[0][1]

            symbol[index] = np.array((x - min_x + padding, y - min_y + padding), dtype=np.int32)


def create_dataset(formulas, dataset_name, encoding):
    object_detection_images = []
    object_detection_bounding_boxes = []
    object_detection_truths = []
    object_detection_truth_ids = []

    symbol_detection_images = []
    symbol_detection_truths = []
    symbol_detection_truth_ids = []

    graphics = g.Graphics()
    augmentor = a.Augmentor()
    # writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, fname + ".pkl"))
    for index, formula in enumerate(formulas):
        progress_bar("Processing images", index, len(formulas))

        formula_id = formula[0]
        tracegroups = query("SELECT * FROM public.tracegroup WHERE formulaid=" + str(formula_id))
        inkml_traces = []
        truths = []
        for tracegroup in tracegroups:
            inkml_tracegroups = []
            tracegroup_id = tracegroup[0]
            traces = query("SELECT * FROM public.trace WHERE tracegroupid=" + str(tracegroup_id))
            for trace in traces:
                inkml_tracegroups.append(trace[2])
            inkml_traces.append(inkml_tracegroups)
            truths.append(convert_truth(tracegroup[2]))

        image, points = graphics.create_image(inkml_traces, True)
        image = 255 - image
        image = augmentor.grayscale(image)
        boxes = calculate_bounding_boxes(points)
        normalized_boxes = normalize_bounding_boxes(boxes, u.w(image), u.h(image))

        object_detection_images.append(image)
        object_detection_bounding_boxes.append(normalized_boxes)
        object_detection_truth_ids.append([encoding[truth] for truth in truths])
        object_detection_truths.append(truths)

        symbols = []
        deleted = 0
        for index, trace_group in enumerate(points):
            try:
                translate_to(trace_group)
                symbol_image, _ = graphics.create_image([trace_group], normalize=False)
                symbol_image = 255 - symbol_image
                symbol_image = augmentor.grayscale(symbol_image)
                symbols.append(symbol_image)
            except ZeroDivisionError:
                print("Error, removing index: {} char: {}".format(index, truths[index - deleted]))
                del truths[index - deleted]
                deleted += 1

        symbol_detection_images += symbols
        symbol_detection_truths += truths
        symbol_detection_truth_ids += [encoding[truth] for truth in truths]

        # cv2.imshow("fullim", image)
        # for i in range(len(symbols)):
        #     image = symbols[i]
        #     truth = truths[i]
        #
        #     cv2.imshow(truth, image)
        #     cv2.moveWindow(truth, 0, 0)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow(truth)

    # Save datasets
    fname = os.path.join(FLAGS.output_path, dataset_name + "_object_detection.pkl")
    with open(fname, 'wb') as file:
        pickle.dump((object_detection_images,
                     object_detection_bounding_boxes,
                     object_detection_truths,
                     object_detection_truth_ids), file)

    fname = os.path.join(FLAGS.output_path, dataset_name + "_symbol_recognition.pkl")
    with open(fname, 'wb') as file:
        pickle.dump((symbol_detection_images,
                     symbol_detection_truths,
                     symbol_detection_truth_ids), file)


def create_encoding(formulas):
    vocabulary = set()

    for index, formula in enumerate(formulas):
        progress_bar("Creating vocabulary", index, len(formulas))
        formula_id = formula[0]
        tracegroups = query("SELECT * FROM public.tracegroup WHERE formulaid=" + str(formula_id))
        for tracegroup in tracegroups:
            vocabulary.add(convert_truth(tracegroup[2]))

    return {truth: index + 1 for index, truth in enumerate(sorted(vocabulary))}


def main(_):
    formulas = query(get_query(True))
    encoding = create_encoding(formulas)

    # write_label_map(encoding)

    create_dataset(formulas, "train", encoding)
    eval_formulas = query(get_query(False))
    create_dataset(eval_formulas, "evaluate", encoding)


if __name__ == '__main__':
    tf.app.run()
