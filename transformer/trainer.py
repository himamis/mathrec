import random
import tensorflow as tf
import numpy as np

from trainer import params
from file_utils import read_pkl
from os import path
from transformer import generator
from transformer import model
from transformer import vocabulary
from trainer.metrics import wer, exp_rate
from utilities import progress_bar
from transformer import model_params


random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)


def create_generators(batch_size=32):
    training = read_pkl(path.join(params.data_base_dir, 'training_data.pkl'))
    training_generator = generator.DataGenerator(training, batch_size)

    validating = read_pkl(path.join(params.data_base_dir, 'validating_data.pkl'))
    validating_generator = generator.DataGenerator(validating, 1)

    return training_generator, validating_generator


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

        return learning_rate


def create_model(transformer_params):
    with tf.name_scope("model"):
        return model.TransformerLatex(transformer_params)


def train_loop(sess, writer, train, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder,
               output_masks_placeholder):
    training, validating = create_generators(params.batch_size)
    global_step = 0

    tf_epoch = tf.Variable(0, dtype=tf.int64, name="epoch")
    with tf.name_scope("evaluation"):
        tf_avg_wer = tf.Variable(1, dtype=tf.float32, name="avg_wer")
        tf_avg_acc = tf.Variable(0, dtype=tf.float32, name="avg_acc")
        tf_avg_exp_rate = tf.Variable(0, dtype=tf.float32, name="avg_exp_rate")

        with tf.contrib.summary.record_summaries_every_n_global_steps(params.epoch_per_validation, tf_epoch):
            tf.contrib.summary.scalar("avg_wer", tf_avg_wer, "validation", tf_epoch)
            tf.contrib.summary.scalar("avg_acc", tf_avg_acc, "validation", tf_epoch)
            tf.contrib.summary.scalar("avg_exp_rate", tf_avg_exp_rate, "validation", tf_epoch)

    if params.start_epoch != -1:
        pass
        # saver.restore(sess, save_format.format(params.start_epoch))
    else:
        tf.global_variables_initializer().run()

    tf.contrib.summary.initialize(graph=tf.get_default_graph())

    for epoch in range(params.start_epoch + 1, params.epochs):
        sess.run(tf.assign(tf_epoch, epoch))
        steps = training.steps()
        for step in range(steps):
            progress_bar("Epoch {}".format(epoch + 1), step + 1, steps)
            encoded_tokens, bounding_boxes, encoded_formulas, encoded_formulas_masks = training.next_batch()
            feed_dict = {
                tokens_placeholder: encoded_tokens,
                bounding_box_placeholder: bounding_boxes,
                output_placeholder: encoded_formulas,
                output_masks_placeholder: encoded_formulas_masks
            }
            # if global_step % summary_step == 0:
            sess.run([train, tf.contrib.summary.all_summary_ops()], feed_dict)
                # writer.add_summary(summary, global_step)
            # else:
                # _ = sess.run([train], feed_dict=feed_dict)
            global_step += 1

        # writer.flush()
        tf.contrib.summary.flush()
        training.reset()

        if (epoch + 1) % params.epoch_per_validation != 0:
            continue

        wern = 0
        exprate = 0
        accn = 0
        for validation_step in range(validating.steps()):
            progress_bar("Validating", validation_step + 1, validating.steps())
            encoded_tokens, bounding_boxes, encoded_formulas, _ = validating.next_batch()
            feed_dict = {
                tokens_placeholder: encoded_tokens,
                bounding_box_placeholder: bounding_boxes,
                output_placeholder: encoded_formulas  # ,
                # output_masks_placeholder: encoded_formulas_masks
            }
            outputs = sess.run(eval_fn, feed_dict)
            result = outputs['outputs'][0]
            target = encoded_formulas[0]
            cwer = wer(result, target) / max(len(target), len(result))
            wern += cwer
            exprate += exp_rate(target, result)
            if abs(cwer) < 1e-6:
                accn += 1

        validating.reset()

        avg_wer = float(wern) / float(validating.steps())
        avg_acc = float(accn) / float(validating.steps())
        avg_exp_rate = float(exprate) / float(validating.steps())

        sess.run([
            tf.assign(tf_avg_wer, avg_wer),
            tf.assign(tf_avg_exp_rate, avg_exp_rate),
            tf.assign(tf_avg_acc, avg_acc)
        ])


def main(transformer_params):
    encoding_vb = vocabulary.encoding_vocabulary
    transformer_params.update(vocab_size=len(encoding_vb))

    tokens_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="tokens")
    bounding_box_placeholder = tf.placeholder(tf.float32, shape=(None, None, 4), name="bounding_boxes")

    output_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="output")
    output_masks_placeholder = tf.placeholder(tf.float32, shape=(None, None), name="output_masks")

    with tf.device(params.device):
        model = create_model(transformer_params)
        logits = model(tokens_placeholder, bounding_box_placeholder, output_placeholder, True)

        # Create loss function
        loss = tf.contrib.seq2seq.sequence_loss(logits, output_placeholder, output_masks_placeholder)
        # L2 regularization
        # decay = 1e-4
        # for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #     print(variable.name)
        #     loss += decay * tf.reduce_sum(tf.pow(variable, 2))

        # Create Optimizer
        learning_rate = get_learning_rate(
            learning_rate=transformer_params["learning_rate"],
            hidden_size=transformer_params["hidden_size"],
            learning_rate_warmup_steps=transformer_params["learning_rate_warmup_steps"])

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=transformer_params["optimizer_adam_beta1"],
            beta2=transformer_params["optimizer_adam_beta2"],
            epsilon=transformer_params["optimizer_adam_epsilon"])
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        # Calculate and apply gradients using LazyAdamOptimizer.
        grads_and_vars = optimizer.compute_gradients(loss)

        # Gradient clipping
        # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        step = tf.train.get_or_create_global_step()
        train = optimizer.apply_gradients(grads_and_vars, global_step=step)

        result = tf.argmax(tf.nn.softmax(logits), output_type=tf.int32, axis=2)
        accuracy = tf.contrib.metrics.accuracy(result, output_placeholder, output_masks_placeholder)

        eval_fn = model(tokens_placeholder, bounding_box_placeholder)

    train_dir = path.join(params.tensorboard_log_dir, params.tensorboard_name)
    writer = tf.contrib.summary.create_file_writer(train_dir)

    with writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
        # Summarization ops
        if params.verbose_summary:
            for grad, var in grads_and_vars:
                tf.contrib.summary.histogram("gradient/" + var.name, grad)
            for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.contrib.summary.histogram("var/" + variable.name, variable)

        tf.contrib.summary.scalar("loss", loss)
        tf.contrib.summary.scalar("accuracy", accuracy)
        tf.contrib.summary.scalar("learning_rate", learning_rate)

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=params.allow_soft_placement)

        with tf.Session(config=config) as sess:
            train_loop(sess, writer, train, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder,
                       output_masks_placeholder)


# main(model_params.BASE_PARAMS)
main(model_params.TINY_PARAMS)
