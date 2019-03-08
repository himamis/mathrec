import tensorflow as tf
import tensorflow.contrib.slim as slim
import trainer.params as params
from object_detection.generator import *


def main():
    training_generator = create_generator(
        "/Users/balazs/Documents/datasets/object_symbol_dataset/evaluate_symbol_recognition.pkl")
    images, labels = create_image_labels(training_generator)
    slim.evaluation.evaluation_loop('', params.tensorboard_log_dir, )