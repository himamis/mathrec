import tensorflow as tf
import object_detection.train_classification


def main(_):
    object_detection.train_classification.main()


if __name__ == '__main__':
    tf.app.run()
