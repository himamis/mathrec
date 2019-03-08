import os
import tensorflow as tf
import slim.nets.resnet_v2 as resnet
import slim.nets.vgg as vgg
from object_detection.generator import create_generator, create_image_labels
import trainer.params as par


def create_input_fn(training=True, batch_size=32):
    generator, obj = create_generator(os.path.join(par.data_base_dir, "training.pkl" if training else "validation.pkl"))
    return create_image_labels(generator, batch_size=batch_size, repeat=None if training else obj.size())


def model_fn(features, labels, mode, params):
    if type == "vgg16":
        logits, _ = vgg.vgg_16(
                tf.to_float(features),
                num_classes=101,
                dropout_keep_prob=1 - params.dropout_rate,
                is_training=mode == tf.estimator.ModeKeys.TRAIN)
    elif type == "resnet50":
        logits, _ = resnet.resnet_v2_50(
            tf.to_float(features),
            num_classes=101,
            is_training=mode == tf.estimator.ModeKeys.TRAIN
        )
    else:
        assert "Type must be available"

    class_ids = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=class_ids, name='acc_op')
    mean_per_class = tf.metrics.mean_per_class_accuracy(labels=labels, predictions=class_ids, num_classes=101)

    metrics = {
        'accuracy': accuracy,
        'mean_per_class_accuracy': mean_per_class
    }
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('mean_per_class_accuracy', mean_per_class[1])

    labels = labels - 1
    labels_one_hot = tf.one_hot(labels, 101, dtype=tf.int32)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_one_hot, logits=logits)
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    else:
        train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': tf.argmax(logits),
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
    else:
        predictions = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def create_estimator(run_config, hparams):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config
    )


def create_train_and_eval_spec(hparams):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: create_input_fn(training=True),
        max_steps=hparams.max)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=create_input_fn(training=False),
        steps=None)
    return train_spec, eval_spec


def main():
    hparams = tf.contrib.training.HParams(
        type='resnet50',
        max_steps=10000,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.6
    )
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=200,
        tf_random_seed=1234567,
        model_dir=os.path.join(par.tensorboard_log_dir, par.tensorboard_name)
    )

    estimator = create_estimator(run_config, hparams)
    train_spec, eval_spec = create_train_and_eval_spec(hparams)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
