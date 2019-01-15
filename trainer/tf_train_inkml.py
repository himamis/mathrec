import file_utils as utils
from trainer import tf_model
from numpy.random import seed
from datetime import datetime
from os import path
import os
from trainer.tf_generator import DataGenerator, TokenDataGenerator
from trainer.tf_predictor import create_predictor
import trainer.default_type as t
import random
import math
from trainer.metrics import wer, exp_rate
from generator import simple_number_operation_generator, Config
from token_parser import Parser
from inkml_graphics import create_graphics_factory
from trainer import params
import time

from tensorflow import set_random_seed
import tensorflow as tf
import numpy as np

random.seed(1337)
seed(1337)
set_random_seed(1337)

parameter_count = True
overfit_testing = False
token_generator = True
measure_time = False
epoch_per_validation = 1

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
weights_fname = 'weights_{epoch}.h5'
history_fname = 'history.pkl'
results_fname = 'results.pkl'
checkpoint_fname = 'checkpoint_epoch_{}.ckpt'


folder_str = folder_str + '-' + params.tensorboard_name
base_dir = path.join(params.model_checkpoint_dir, folder_str)
save_format = path.join(base_dir, checkpoint_fname)
if params.gcs is not None:
    save_format = os.path.join("gs://{}".format(params.gcs), save_format)
if params.ckpt_dir is not None:
    save_format = os.path.join(params.ckpt_dir, checkpoint_fname)

#if not path.exists(base_dir):
#    os.mkdir(base_dir)

start_time = datetime.now()

batch_size = 6
step_per_summary = int(math.ceil(100 / batch_size))
epochs = 600
# levels = 5
decay = 1e-4
encoding_vb, decoding_vb = utils.read_pkl(path.join(params.data_base_dir, "vocabulary.pkl"))
decoding_vb = {k: v for k, v in decoding_vb.items() if v != "<start>"}

image, truth, _ = zip(*utils.read_pkl(path.join(params.data_base_dir, "data_train.pkl")))
image_valid, truth_valid, _ = zip(*utils.read_pkl(path.join(params.data_base_dir, "data_validate.pkl")))

generator = DataGenerator(image, truth, encoding_vb, batch_size=batch_size)
generator_valid = DataGenerator(image_valid, truth_valid, encoding_vb, 1)

if overfit_testing:
    image, truth, _ = zip(*utils.read_pkl(path.join(params.data_base_dir, "overfit.pkl")))
    new_vocab = "1234567890-+"
    new_vocab = list(new_vocab)
    new_vocab.append("<end>")
    new_vocab.append("<start>")

    encoding_vb = dict(zip(new_vocab, range(len(new_vocab))))
    decoding_vb = { v: k for k, v in encoding_vb.items() if k != "<start>"}

    image_valid = image
    truth_valid = truth

    if token_generator:
        gen = simple_number_operation_generator()
        conf = Config()
        parser = Parser(create_graphics_factory(os.path.join(params.data_base_dir, 'tokengroup.pkl')))
        generator = TokenDataGenerator(gen, parser, conf, encoding_vb, batch_size, 10)
    else:
        generator = DataGenerator(image, truth, encoding_vb, batch_size)
    generator_valid = DataGenerator(image, truth, encoding_vb, 1)

image_width = None
image_height = None
batch_size = None

pl_input_images = tf.placeholder(t.my_tf_float, shape=(batch_size, image_width, image_height, 1), name="input_images")
pl_image_masks = tf.placeholder(t.my_tf_float, shape=(batch_size, image_width, image_height, 1), name="input_image_masks")
pl_input_characters = tf.placeholder(tf.int32, shape=(batch_size, None), name="input_characters")
pl_is_training = tf.placeholder(tf.bool, name="is_training")

pl_r_max = tf.placeholder(t.my_tf_float, name="r_max", shape=())
pl_d_max = tf.placeholder(t.my_tf_float, name="d_max", shape=())

print("Image2Latex: Start create model: {}".format(str(datetime.now().time())))
device = '/cpu:0' if params.use_gpu == 'n' else '/gpu:{}'.format(params.use_gpu)
with tf.device(device):
    # tf.summary.image("input_images", pl_input_images)

    model = tf_model.Model(len(encoding_vb),
                           # filter_sizes=[32, 64],
                           encoder_size=256,
                           decoder_units=512,
                           attention_dim=512,
                           embedding_dim=512,
                           # bidirectional=True,
                           conv_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           conv_bias_init=tf.initializers.random_normal(),
                           conv_activation=tf.nn.relu,
                           cnn_block=tf_model.default_cnn_block,
                           encoder_kernel_init=tf.initializers.random_normal(dtype=t.my_tf_float),
                           decoder_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           decoder_bias_init=tf.initializers.constant(1/4, dtype=t.my_tf_float),
                           dense_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           dense_bias_init=tf.initializers.zeros(dtype=t.my_tf_float),
                           decoder_recurrent_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float))
    eval_feature_grid, eval_masking = model.feature_grid(pl_input_images, pl_image_masks, is_training=pl_is_training,
                                                         summarize=False, r_max=pl_r_max, d_max=pl_d_max)
    eval_calculate_h0, eval_calculate_alphas = model.calculate_decoder_init(eval_feature_grid, eval_masking)
    output, (states_h, states_alpha) = model.decoder(eval_feature_grid, eval_masking, pl_input_characters,
                                                     eval_calculate_h0, eval_calculate_alphas, summarize=True,
                                                     input_images=pl_input_images)

    eval_output_softmax = tf.nn.softmax(output)

if parameter_count:
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total parameter num: {}".format(total_parameters))

print("Image2Latex: End create model: {}".format(str(datetime.now().time())))

pl_y_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size, None), name="y_labels")
pl_sequence_masks = tf.placeholder(dtype=t.my_tf_float, shape=(batch_size, None), name="y_labels_masks")

with tf.device(device):
    with tf.name_scope("loss"):
        loss = tf.contrib.seq2seq.sequence_loss(output, pl_y_tensor, pl_sequence_masks)

        # L2 regularization
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if not "batch_norm" in variable.name:
                loss += decay * tf.reduce_sum(tf.pow(variable, 2))

with tf.device('/cpu:0'):
    tf.summary.scalar("loss", loss)

    if params.verbose_summary:
        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            tf.summary.histogram(variable.name, variable)

with tf.device(device):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(loss)

    with tf.device('/cpu:0'):
        if params.verbose_summary:
            for grad, var in grads_and_vars:
                tf.summary.histogram("gradient/" + var.name, grad)

    # Gradient clipping
    grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = optimizer.apply_gradients(grads_and_vars)

    with tf.name_scope("accuracy"):
        result = tf.argmax(tf.nn.softmax(output), output_type=tf.int32, axis=2)

        accuracy = tf.contrib.metrics.accuracy(result, pl_y_tensor, pl_sequence_masks)

        with tf.device('/cpu:0'):
            tf.summary.scalar("accuracy", accuracy)

with tf.device('/cpu:0'):
    saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
no_summary_per_epoch = 40
summary_step = max(math.floor(generator.steps() / no_summary_per_epoch), 1)
patience = 10
best_exp_rate = -1

valid_avg_wer_summary = tf.Summary()
valid_avg_acc_summary = tf.Summary()
valid_avg_exp_rate_summary = tf.Summary()
level_summary = tf.Summary()

valid_avg_wer_summary.value.add(tag="valid_avg_wer", simple_value=None)
valid_avg_acc_summary.value.add(tag="valid_avg_acc", simple_value=None)
valid_avg_exp_rate_summary.value.add(tag="valid_exp_rate", simple_value=None)
level_summary.value.add(tag="level", simple_value=None)


def main_training(sess: tf.Session, pctx, opts):
    """ Main function that runs the training"""

    # Initialize some variables
    global_step = 1
    r_max_val_init = 1
    d_max_val_init = 0
    r_max_val = r_max_val_init
    d_max_val = d_max_val_init
    bad_counter = 0
    best_wer = 999999

    if params.start_epoch != 0:
        saver.restore(sess, save_format.format(params.start_epoch))
    else:
        tf.global_variables_initializer().run()
    predictor = create_predictor(sess,
                                 (pl_input_images, pl_image_masks, pl_is_training, pl_r_max, pl_d_max),
                                 (eval_feature_grid, eval_masking, eval_calculate_h0, eval_calculate_alphas),
                                 (pl_input_characters),
                                 (eval_output_softmax, states_h, states_alpha),
                                 encoding_vb, decoding_vb, k=100, max_length=100)

    writer = tf.summary.FileWriter(os.path.join(params.tensorboard_log_dir, params.tensorboard_name))
    writer.add_graph(sess.graph)

    for epoch in range(params.start_epoch, epochs):
        print("Staring epoch {}".format(epoch + 1))

        generator.reset()
        for step in range(generator.steps()):

            # Generate same lengths per batch
            if overfit_testing and token_generator:
                length = np.random.randint(1, 7)
                gen.min_length = length
                gen.max_length = length + 1

            # Measure runtime
            if measure_time:
                start_time = time.time()

            image, label, observation, masks, label_masks = generator.next_batch()

            if measure_time:
                print("Data fetch \t %s \t seconds" % (time.time() - start_time))

            # import cv2
            # cv2.imshow("image", image[0])
            # cv2.waitKey(0)

            feed_dict = {
                pl_input_images: image,
                pl_input_characters: observation,
                pl_sequence_masks: label_masks,
                pl_image_masks: masks,
                pl_y_tensor: label,
                pl_is_training: True,
                pl_r_max: r_max_val,
                pl_d_max: d_max_val
            }

            if measure_time:
                start_time = time.time()

            # Enable tracing for next session.run.
            pctx.trace_next_step()
            # Dump the profile after the step.
            pctx.dump_next_step()

            if global_step % summary_step == 0:
                vloss, vacc, s, _ = sess.run([loss, accuracy, merged_summary, train], feed_dict=feed_dict)
                writer.add_summary(s, global_step)
            else:
                vloss, vacc, _ = sess.run([loss, accuracy, train], feed_dict=feed_dict)

            pctx.profiler.profile_operations(options=opts)

            if measure_time:
                print("Training step \t %s \t seconds" % (time.time() - start_time))

            print("Loss: {}, Acc: {}".format(vloss, vacc))

            from_epoch = 40
            until_epoch = 80
            diff = max(min((epoch - from_epoch) / (until_epoch - from_epoch), 1), 0)
            r_max_val = r_max_val_init + 2 * diff
            d_max_val = d_max_val_init + 5 * diff
            print("Step {}: r_max_val {}, d_max_val {}".format(global_step, r_max_val, d_max_val))
            global_step += 1

        # Validation after each epoch
        if (epoch + 1) % epoch_per_validation != 0:
            continue
        wern = 0
        accn = 0
        exprate = 0
        for step in range(generator_valid.steps()):
            print("Validation step {}".format(step))
            image, label, observation, masks, lengths = generator_valid.next_batch()
            predict = predictor(image, masks)
            re_encoded = [encoding_vb[s] for s in predict]
            target = label[0][:-1]
            cwer = wer(re_encoded, target) / max(len(target), len(re_encoded))
            wern += cwer
            exprate += exp_rate(target, re_encoded)
            if abs(cwer) < 1e-6:
                accn += 1

        avg_wer = float(wern) / float(generator_valid.steps())
        avg_acc = float(accn) / float(generator_valid.steps())
        avg_exp_rate = float(exprate) / float(generator_valid.steps())

        print("Avg_wer: {}, avg_acc: {}, avg_exp_rate: {}".format(avg_wer, avg_acc, avg_exp_rate))

        valid_avg_wer_summary.value[0].simple_value = avg_wer
        valid_avg_acc_summary.value[0].simple_value = avg_acc
        valid_avg_exp_rate_summary.value[0].simple_value = avg_exp_rate
        if writer is not None:
            writer.add_summary(valid_avg_wer_summary, epoch)
            writer.add_summary(valid_avg_acc_summary, epoch)
            writer.add_summary(valid_avg_exp_rate_summary, epoch)
            writer.flush()

        improved = False

        # if avg_exp_rate > best_exp_rate:
        #     best_exp_rate = avg_exp_rate
        #     improved = True

        if avg_wer < best_wer:
            best_wer = avg_wer
            improved = True

        if improved:
            print("Epoch {}: Improved".format(epoch))
            bad_counter = 0
            saver.save(sess, save_format.format(epoch))
        else:
            bad_counter += 1
            print("Epoch {}: Not improved, bad counter: {}".format(epoch, bad_counter))

        if bad_counter == patience:
            print("Early stopping")
            break

        generator_valid.reset()


print("Image2Latex Start training...")

builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()
with tf.contrib.tfprof.ProfileContext(params.profiling, enabled=params.profiling != 'n') as pctx:
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=params.allow_soft_placement)
    with tf.Session(config=config) as sess:
        main_training(sess, pctx, opts)

