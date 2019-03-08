import tensorflow as tf
import keras
import pickle
import slim.nets.resnet_v2 as resnet
import tensorflow.contrib.slim as slim
from object_detection.generator import create_generator, create_image_labels
from tensorflow.contrib.slim.python.slim.learning import train_step


import trainer.params as params

validation_every_n_step = 400


def main():
    training_generator = create_generator("/Users/balazs/Documents/datasets/object_symbol_dataset/train_symbol_recognition.pkl")
    images, labels = create_image_labels(training_generator)

    validation_generator = create_generator("/Users/balazs/Documents/datasets/object_symbol_dataset/evaluate_symbol_recognition.pkl")
    validate_images, validate_labels = create_image_labels(validation_generator, shuffle=False)

    tf.contrib.summary.image('images/input', images)
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        predictions, _ = resnet.resnet_v2_50(
            tf.to_float(images),
            num_classes=101,
            is_training=True,
        )

        validate_predictions = resnet.resnet_v2_50(
            tf.to_float(validate_images),
            num_classes=101,
            is_training=False
        )

    labels = labels - 1
    labels_one_hot = tf.one_hot(labels, 101, dtype=tf.int32)

    # slim.losses.add_loss(tf.losses.softmax_cross_entropy(net, pl_labels))
    slim.losses.softmax_cross_entropy(predictions, labels_one_hot)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Summaries
    accuracy = slim.metrics.accuracy(tf.nn.softmax(predictions), labels_one_hot)
    tf.summary.scalar('accuracy/accuracy', accuracy)

    validate_labels = validate_labels - 1
    validate_labels_one_hot = tf.one_hot(validate_labels, 101)
    validate_accuracy = slim.metrics.accuracy(tf.nn.softmax(validate_predictions), validate_labels_one_hot)

    def train_step_fn(session, *args, **kwargs):
        loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % validation_every_n_step == 0:
            accuracy = session.run(validate_accuracy)
            print('Accuracy {}'.format(accuracy))

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0

    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    slim.learning.train(train_op, params.tensorboard_log_dir,
                        train_step_fn=train_step_fn,
                        number_of_steps=50000000, save_summaries_secs=300, save_interval_secs=600)
    print("Done")
