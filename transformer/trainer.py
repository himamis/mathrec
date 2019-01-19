import math
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


random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)


def create_generators(batch_size=32):
    training = read_pkl(path.join(params.data_base_dir, 'training_data.pkl'))
    training_generator = generator.DataGenerator(training, batch_size)

    testing = read_pkl(path.join(params.data_base_dir, 'testing_data.pkl'))
    testing_generator = generator.DataGenerator(testing, 1)

    return training_generator, testing_generator


def create_model():
    encoding_vb = vocabulary.encoding_vocabulary
    return model.TransformerLatex(len(encoding_vb))


def train_loop(sess, train, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder, output_masks_placeholder):
    training, testing = create_generators(params.batch_size)
    no_summary_per_epoch = 40
    summary_step = max(math.floor(training.steps() / no_summary_per_epoch), 1)
    global_step = 0

    # Create writer
    writer = tf.summary.FileWriter(path.join(params.tensorboard_log_dir, params.tensorboard_name))
    writer.add_graph(sess.graph)

    if params.start_epoch != -1:
        pass
        # saver.restore(sess, save_format.format(params.start_epoch))
    else:
        tf.global_variables_initializer().run()
    merged_summary = tf.summary.merge_all()

    valid_avg_wer_summary = tf.Summary()
    valid_avg_acc_summary = tf.Summary()
    valid_avg_exp_rate_summary = tf.Summary()

    valid_avg_wer_summary.value.add(tag="valid_avg_wer", simple_value=None)
    valid_avg_acc_summary.value.add(tag="valid_avg_acc", simple_value=None)
    valid_avg_exp_rate_summary.value.add(tag="valid_exp_rate", simple_value=None)

    for epoch in range(params.start_epoch + 1, params.epochs):
        print("Staring epoch {}".format(epoch + 1))
        for step in range(training.steps()):
            encoded_tokens, bounding_boxes, encoded_formulas, encoded_formulas_masks = training.next_batch()
            feed_dict = {
                tokens_placeholder: encoded_tokens,
                bounding_box_placeholder: bounding_boxes,
                output_placeholder: encoded_formulas,
                output_masks_placeholder: encoded_formulas_masks
            }
            if global_step % summary_step == 0:
                summary, _ = sess.run([merged_summary, train], feed_dict)
                writer.add_summary(summary, global_step)
            else:
                _ = sess.run([train], feed_dict=feed_dict)

            global_step += 1

        training.reset()

        wern = 0
        exprate = 0
        accn = 0
        for validation_step in range(testing.steps()):
            print("Validation step {}".format(validation_step + 1))
            encoded_tokens, bounding_boxes, encoded_formulas, _ = testing.next_batch()
            feed_dict = {
                tokens_placeholder: encoded_tokens,
                bounding_box_placeholder: bounding_boxes,
                output_placeholder: encoded_formulas # ,
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

        testing.reset()

        avg_wer = float(wern) / float(testing.steps())
        avg_acc = float(accn) / float(testing.steps())
        avg_exp_rate = float(exprate) / float(testing.steps())

        valid_avg_wer_summary.value[0].simple_value = avg_wer
        valid_avg_acc_summary.value[0].simple_value = avg_acc
        valid_avg_exp_rate_summary.value[0].simple_value = avg_exp_rate
        writer.add_summary(valid_avg_wer_summary, epoch)
        writer.add_summary(valid_avg_acc_summary, epoch)
        writer.add_summary(valid_avg_exp_rate_summary, epoch)
        writer.flush()



def main():
    encoding_vb = vocabulary.encoding_vocabulary

    tokens_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="tokens")
    bounding_box_placeholder = tf.placeholder(tf.float32, shape=(None, None, 4), name="bounding_boxes")

    output_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="output")
    output_masks_placeholder = tf.placeholder(tf.float32, shape=(None, None), name="output_masks")

    with tf.device(params.device):
        model = create_model()
        logits = model(tokens_placeholder, bounding_box_placeholder, output_placeholder, True)

        # Create loss function
        loss = tf.contrib.seq2seq.sequence_loss(logits, output_placeholder, output_masks_placeholder)

        # Create Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = optimizer.compute_gradients(loss)

        # Gradient clipping
        # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        train = optimizer.apply_gradients(grads_and_vars)

        result = tf.argmax(tf.nn.softmax(logits), output_type=tf.int32, axis=2)
        accuracy = tf.contrib.metrics.accuracy(result, output_placeholder, output_masks_placeholder)

        eval_fn = model(tokens_placeholder, bounding_box_placeholder)

    # Summarization ops
    if params.verbose_summary:
        for grad, var in grads_and_vars:
            tf.summary.histogram("gradient/" + var.name, grad)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=params.allow_soft_placement)
    with tf.Session(config=config) as sess:
        train_loop(sess, train, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder,
                   output_masks_placeholder)


main()