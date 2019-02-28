import tensorflow as tf
from database_create.methods import *
from utilities import parse_arg, progress_bar
import pickle
from inkml import graphics as g
import cv2
import numpy as np
import graphics.utils as u
import os

from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2 as silm

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


def create_tf_example(example):
    image = example['image']
    # TODO(user): Populate the following variables from your example.

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

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def calculate_bounding_boxes(inkml, width, height):
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

    scalex = float(width)
    scaley = float(height)

    # Normalize bounding boxes
    for index, box in enumerate(bounding_boxes):
        bounding_boxes[index] = (
            float(box[0]) / scalex,
            float(box[1]) / scaley,
            float(box[2]) / scalex,
            float(box[3]) / scaley
        )

    return bounding_boxes


def draw_rectangles(image, boxes):
    #for box in boxes:
    #    cv2.rectangle(image, (
    #        box[0] * float(u.w(image))), box[1]), (box[2], box[3]), (0, 255, 0), 1)
    pass


def main(_):
    formulas = query("select formula.id, formula.writerid, formula.formula FROM public.database, public.writer, "
                     "public.formula WHERE public.formula.writerid = public.writer.id "
                     "AND public.writer.databaseid = public.database.id "
                     "AND public.database.name NOT LIKE 'CROHME2016_data/Test2016_INKML_GT';")
    vocabulary = set()

    for index, formula in enumerate(formulas):
        progress_bar("Creating vocabulary", index, len(formulas))
        formula_id = formula[0]
        tracegroups = query("SELECT * FROM public.tracegroup WHERE formulaid=" + str(formula_id))
        for tracegroup in tracegroups:
            vocabulary.add(tracegroup[2])

    print(vocabulary)
    encoding = {truth: index + 1 for index, truth in enumerate(sorted(vocabulary))}
    label_map = create_label_map(encoding)
    with open(os.path.join(FLAGS.output_path, "dataset_label_map.pbtxt"), 'w') as file:
        file.write(label_map)

    graphics = g.Graphics()
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, "dataset.proto"))
    for index, formula in enumerate(formulas):
        progress_bar("Processing images", index, len(formulas))

        formula_id = formula[0]
        tracegroups = query("SELECT * FROM public.tracegroup WHERE formulaid=" + str(formula_id))
        inkml_traces = []
        truths = []
        for tracegroup in tracegroups:
            inkml_tracegroups = []
            tracegroup_id = tracegroup[0]
            vocabulary.add(tracegroup[2])
            traces = query("SELECT * FROM public.trace WHERE tracegroupid=" + str(tracegroup_id))
            for trace in traces:
                inkml_tracegroups.append(trace[2])
            inkml_traces.append(inkml_tracegroups)
            truths.append(tracegroup[2])

        image, points = graphics.create_image(inkml_traces, True)
        boxes = calculate_bounding_boxes(points, u.w(image), u.h(image))
        # draw_rectangles(image, boxes)
        image = 255 - image

        encoded_image = cv2.imencode('.png', image)[1].tobytes()
        tf_example = create_tf_example({
            'image': image,
            'encoded_image': encoded_image,
            'format': b'png',
            'bounding_boxes': boxes,
            'truths': [truth.encode('utf-8') for truth in truths],
            'truth_ids': [encoding[truth] for truth in truths]
        })
        writer.write(tf_example.SerializeToString())
        # cv2.imshow("image", image)
        # cv2.waitKey(0)

    # examples = []
    # TODO(user): Write code to read in your dataset to examples variable

    #for example in examples:
    #    tf_example = create_tf_example(example)
    #    writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
