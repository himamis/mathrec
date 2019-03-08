import os
import tensorflow as tf
#import slim.nets.resnet_v2 as resnet
import slim.nets.vgg as vgg
from object_detection.generator import create_generator, create_image_labels


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
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_one_hot, logits=logits)
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
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(params.tensorboard_log_dir, params.tensorboard_name)
    )

    train_input_fn, _ = create_input_fn(training=True)
    eval_input_fn, obj = create_input_fn(training=False)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=500000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=None)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)