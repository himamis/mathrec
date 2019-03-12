import os
import tensorflow as tf
import slim.nets.resnet_v2 as resnet
import slim.nets.vgg as vgg
from object_detection.generator import create_dataset_tensors
import trainer.params as par


def create_input_fn(training=True, batch_size=32, epochs=20):
    ds = create_dataset_tensors(
        os.path.join(par.data_base_dir, "training.pkl" if training else "validation.pkl"),
        batch_size=batch_size, repeat=None, shuffle=training
    )
    if training:
        ds = ds.repeat()

    return ds


def model_fn(features, labels, mode, params):
    if params.type == "vgg16":
        logits, _ = vgg.vgg_16(
            tf.to_float(features),
            num_classes=101,
            dropout_keep_prob=1 - params.dropout_rate,
            is_training=mode == tf.estimator.ModeKeys.TRAIN)
    elif params.type == "vgg19":
        logits, _ = vgg.vgg_19(
            tf.to_float(features),
            num_classes=101,
            dropout_keep_prob=1 - params.dropout_rate,
            is_training=mode == tf.estimator.ModeKeys.TRAIN)
    elif params.type == "resnet50":
        logits, _ = resnet.resnet_v2_50(
            tf.to_float(features),
            num_classes=101,
            is_training=mode == tf.estimator.ModeKeys.TRAIN
        )
    else:
        raise ValueError("type not available")

    class_ids = tf.argmax(logits, 1, output_type=tf.int32)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=class_ids)
    else:
        labels = labels - 1
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy_metric = tf.metrics.accuracy(labels=labels, predictions=class_ids)

        equality = tf.equal(class_ids, labels)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "train_accuracy": accuracy}, every_n_iter=100)

        metrics = {
            'accuracy': accuracy_metric,
        }

        if mode == tf.estimator.ModeKeys.TRAIN:
            #optimizer = tf.train.AdamOptimizer(
            #    learning_rate=params.learning_rate
            #)
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=params.learning_rate
            )
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        else:
            train_op = None
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=metrics,
                                          training_hooks=[logging_hook])
    return spec


def create_estimator(run_config, hparams):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config
    )


def create_train_and_eval_spec(hparams):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: create_input_fn(training=True, epochs=None),
        max_steps=hparams.max_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: create_input_fn(training=False, epochs=1),
        steps=None)
    return train_spec, eval_spec


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    hparams = tf.contrib.training.HParams(
        type='vgg19',
        # epochs=epochs,
        max_steps=200000,
        num_epochs=200,
        batch_size=32,
        learning_rate=0.01,
        dropout_rate=0.6,
        momentum=0.9
    )
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=200,
        tf_random_seed=1234567,
        model_dir=os.path.join(par.tensorboard_log_dir, par.tensorboard_name)
    )

    estimator = create_estimator(run_config, hparams)
    train_spec, eval_spec = create_train_and_eval_spec(hparams)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print("Done")
