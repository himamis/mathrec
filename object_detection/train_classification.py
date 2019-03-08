import os
import tensorflow as tf
#import slim.nets.resnet_v2 as resnet
import slim.nets.vgg as vgg
import tensorflow.contrib.slim as slim
from object_detection.generator import create_generator, create_image_labels
from tensorflow.contrib.slim.python.slim.learning import train_step


import trainer.params as params

validation_every_n_step = 10
decay = 1e-4
batch_size = 32


def create_input_fn(training=True):
    generator, obj = create_generator(os.path.join(params.data_base_dir, "training.pkl" if training else "validation.pkl"))
    return lambda: create_image_labels(generator, batch_size=batch_size, repeat=50 if training else 1), obj


def model_fn(features, labels, mode, params):
    logits, _ = vgg.vgg_16(
            tf.to_float(features),
            num_classes=101,
            dropout_keep_prob=0.4,
            is_training=mode == tf.estimator.ModeKeys.TRAIN)
    labels = labels - 1
    labels_one_hot = tf.one_hot(labels, 101, dtype=tf.int32)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1), name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    else: # tf.estimator.ModeKeys.EVAL
        predictions = {
            'class_ids': tf.argmax(logits),
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def main():
    #data_base_dir = params.data_base_dir
    #training_generator, _ = create_generator(os.path.join(data_base_dir, "training.pkl"))
    #images, labels = create_image_labels(training_generator, batch_size=batch_size)

    # tf.contrib.summary.image('images/input', images)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(params.tensorboard_log_dir, params.tensorboard_name)
    )

    train_input_fn, _ = create_input_fn(training=True)
    eval_input_fn, obj = create_input_fn(training=False)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=500000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=None)
    # classifier.train(input_fn=train_input_fn, steps=20000)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    #
    #
    #
    #
    #
    # with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #     predictions, _ = vgg.vgg_16(
    #         tf.to_float(images),
    #         num_classes=101,
    #         dropout_keep_prob=0.4,
    #         is_training=True
    #     )
    #     # predictions, _ = #resnet.resnet_v2_50(
    #     #     tf.to_float(images),
    #     #     num_classes=101,
    #     #     is_training=True,
    #     # )
    #
    #     validate_predictions, _ = vgg.vgg_16(
    #         tf.to_float(validate_images),
    #         num_classes=101,
    #         dropout_keep_prob=0.4,
    #         is_training=False
    #     )
    #
    # labels = labels - 1
    # labels_one_hot = tf.one_hot(labels, 101, dtype=tf.int32)
    #
    # # slim.losses.add_loss(tf.losses.softmax_cross_entropy(net, pl_labels))
    # slim.losses.softmax_cross_entropy(predictions, labels_one_hot)
    # total_loss = slim.losses.get_total_loss()
    # # L2 regularization
    # for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #     if not "batch_norm" in variable.name:
    #         total_loss += decay * tf.reduce_sum(tf.pow(variable, 2))
    # tf.summary.scalar('losses/total_loss', total_loss)
    #
    # # Summaries
    # accuracy = slim.metrics.accuracy(tf.argmax(predictions, output_type=tf.int32), labels_one_hot)
    # tf.summary.scalar('accuracy/accuracy', accuracy)
    #
    # validate_labels = validate_labels - 1
    # validate_labels_one_hot = tf.one_hot(validate_labels, 101, dtype=tf.int32)
    # validate_accuracy = slim.metrics.accuracy(tf.argmax(validate_predictions, output_type=tf.int32),
    #                                           validate_labels_one_hot)
    #
    # global_step = tf.train.get_or_create_global_step()
    #
    # def train_step_fn(session, *args, **kwargs):
    #     loss, should_stop = train_step(session, *args, **kwargs)
    #
    #     if session.run(global_step) % validation_every_n_step == 0:
    #         accuracy = 0
    #         for i in range(valid_obj.size()):
    #             accuracy += session.run(validate_accuracy)
    #         print('Accuracy {}'.format(accuracy / valid_obj.size()))
    #
    #     # if train_step_fn.step % validation_every_n_step == 0:
    #     #     accuracy = session.run(validate_accuracy)
    #     #     print('Accuracy {}'.format(accuracy))
    #
    #     # train_step_fn.step += 1
    #     return [total_loss, should_stop]
    #
    # train_step_fn.step = 0
    #
    # tf.estimator.Estimator()
    #
    # optimizer = tf.train.AdamOptimizer(0.001)
    # train_op = slim.learning.create_train_op(total_loss, optimizer)
    # slim.learning.train(train_op, os.path.join(params.tensorboard_log_dir, params.tensorboard_name),
    #                     train_step_fn=train_step_fn,
    #                     number_of_steps=500000,
    #                     save_summaries_secs=60,
    #                     save_interval_secs=600)
    # print("Done")
