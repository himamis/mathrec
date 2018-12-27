import file_utils as utils
from trainer import tf_model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path
import os
from trainer.tf_generator import DataGenerator
from trainer.tf_predictor import create_predictor
import trainer.default_type as t
import random
import math

from tensorflow import set_random_seed
import tensorflow as tf

random.seed(1337)
seed(1337)
set_random_seed(1337)

parameter_count = True

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
weights_fname = 'weights_{epoch}.h5'
history_fname = 'history.pkl'
results_fname = 'results.pkl'
checkpoint_fname = 'checkpoint_epoch_{}.ckpt'

gcs = parse_arg('--gcs', required=False)
use_gpu = parse_arg('--gpu', default='n', required=False)
start_epoch = int(parse_arg('--start-epoch', -1))
ckpt_dir = parse_arg('--ckpt-dir', None, required=False)
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/new_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/tf_model')
tensorboard_log_dir = parse_arg('--tb', None, required=False)
tensorboard_name = parse_arg('--tbn', "adam", required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
save_format = path.join(base_dir, checkpoint_fname)
if gcs is not None:
    save_format = os.path.join("gs://{}".format(gcs), save_format)
if ckpt_dir is not None:
    save_format = os.path.join(ckpt_dir, checkpoint_fname)

#if not path.exists(base_dir):
#    os.mkdir(base_dir)


start_time = datetime.now()
git_hexsha = parse_arg('--git-hexsha', 'NAN')

batch_size = 4
epochs = 200
levels = 5
decay = 1e-4
encoding_vb, decoding_vb = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))

image, truth, _ = zip(*utils.read_pkl(path.join(data_base_dir, "overfit.pkl")))
if True:
    new_vocab = "1234567890-+"
    new_vocab = list(new_vocab)
    new_vocab.append("<end>")
    new_vocab.append("<start>")

    encoding_vb = dict(zip(new_vocab, range(len(new_vocab))))
    decoding_vb = { v: k for k, v in encoding_vb.items() if k != "<start>"}

generator = DataGenerator(image, truth, encoding_vb, batch_size=batch_size)

image_valid, truth_valid, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_validate.pkl")))
generator_valid = DataGenerator(image_valid, truth_valid, encoding_vb, 1)

input_images = tf.placeholder(t.my_tf_float, shape=(batch_size, None, None, 1), name="input_images")
image_masks = tf.placeholder(t.my_tf_float, shape=(batch_size, None, None, 1), name="input_image_masks")
input_characters = tf.placeholder(tf.int32, shape=(batch_size, None), name="input_characters")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

single_input_image = tf.placeholder(t.my_tf_float, shape=(1, None, None, 1), name="input_images")
single_image_mask = tf.placeholder(t.my_tf_float, shape=(1, None, None, 1), name="input_image_masks")
single_input_character = tf.placeholder(tf.int32, shape=(1, None), name="input_characters")


print("Image2Latex: Start create model: {}".format(str(datetime.now().time())))
device = '/cpu:0' if use_gpu == 'n' else '/gpu:{}'.format(use_gpu)
with tf.device(device):
    model = tf_model.Model(len(encoding_vb),
                           filter_sizes=[32, 64, 64, 128],
                           encoder_size=128,
                           decoder_units=512,
                           attention_dim=512,
                           embedding_dim=512,
                           bidirectional=True,
                           conv_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           conv_bias_init=tf.initializers.random_normal(),
                           conv_activation=tf.nn.relu,
                           cnn_block=tf_model.dense_cnn_block_creator(4, 0.2),
                           encoder_kernel_init=tf.initializers.random_normal(dtype=t.my_tf_float),
                           decoder_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           decoder_bias_init=tf.initializers.constant(1/4, dtype=t.my_tf_float),
                           dense_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           dense_bias_init=tf.initializers.zeros(dtype=t.my_tf_float),
                           decoder_recurrent_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float))
    # Training
    training_output = model.training(input_images, image_masks, input_characters)

    # Evaluating
    eval_feature_grid, eval_masking = model.feature_grid(single_input_image, single_image_mask, True)
    eval_calculate_h0 = model.calculate_decoder_init(eval_feature_grid, eval_masking)
    eval_init_h = model.decoder_init(1)
    eval_state_h, eval_output = model.decoder(eval_feature_grid, eval_masking,
                                                            single_input_character, eval_init_h)
    eval_output_softmax = tf.nn.softmax(eval_output)

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

y_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size, None), name="y_labels")
#lengts_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name="lengths")
sequence_masks = tf.placeholder(dtype=t.my_tf_float, shape=(batch_size, None))

with tf.name_scope("loss"):
    #sequence_masks = tf.sequence_mask(lengts_tensor, dtype=t.my_tf_float)
    tf.summary.histogram("before_softmax", training_output)
    loss = tf.contrib.seq2seq.sequence_loss(training_output, y_tensor, sequence_masks)

    # L2 regularization
    #for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #    if not variable.name.startswith('batch_norm'):
    #        loss += decay * tf.reduce_sum(tf.pow(variable, 2))

    tf.summary.scalar("loss", loss)

lr = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

with tf.name_scope("train"):
    #optimizer = tf.train.GradientDescentOptimizer()
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #optimizer = tf.train.GradientDescentOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss)
    #grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    tf.summary.merge(
        [tf.summary.histogram("gradient-{}".format(g[1].name), g[0]) for g in grads_and_vars if g[0] is not None])
    train = optimizer.apply_gradients(grads_and_vars)

with tf.name_scope("accuracy"):
    result = tf.argmax(tf.nn.softmax(training_output), output_type=tf.int32, axis=2)

    accuracy = tf.contrib.metrics.accuracy(result, y_tensor, sequence_masks)
    accuracy = tf.Print(accuracy, [result, y_tensor, sequence_masks], "Res, Tens, Maks", summarize=20)
    tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
no_summary_per_epoch = 2
summary_step = math.floor(generator.steps() / no_summary_per_epoch)
patience = 5
bad_counter = 0
best_wer = 999999
best_exp_rate = -1
#level = 0
level = 4

lr_val = 0.1
epoch_lr_decay = 50
decay_rate = 0.5

valid_avg_wer_summary = tf.Summary()
valid_avg_acc_summary = tf.Summary()
valid_avg_exp_rate_summary = tf.Summary()
level_summary = tf.Summary()
lr_summary = tf.Summary()

valid_avg_wer_summary.value.add(tag="valid_avg_wer", simple_value=None)
valid_avg_acc_summary.value.add(tag="valid_avg_acc", simple_value=None)
valid_avg_exp_rate_summary.value.add(tag="valid_exp_rate", simple_value=None)
level_summary.value.add(tag="level", simple_value=None)
lr_summary.value.add(tag="learning_rate", simple_value=None)
lr_summary.value[0].simple_value = lr_val

init = tf.global_variables_initializer()

print("Image2Latex Start training...")
global_step = 1

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
with tf.Session(config=config) as sess:
    if start_epoch != -1:
        saver.restore(sess, save_format.format(start_epoch))
        start_epoch = -1
    predictor = create_predictor(sess, (single_input_image, single_image_mask, eval_init_h, eval_feature_grid, eval_masking,
                                 single_input_character), (eval_feature_grid, eval_masking, eval_calculate_h0, eval_output_softmax,
                                                     eval_state_h), encoding_vb, decoding_vb, k=10)
    writer = None
    if tensorboard_log_dir is not None:
        writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, tensorboard_name))
        writer.add_graph(sess.graph)
        writer.add_summary(lr_summary)
    sess.run(init)
    for epoch in range(epochs):

        print("Current level {}".format(level))
        level_summary.value[0].simple_value = level
        if writer is not None:
            writer.add_summary(level_summary, global_step)

        generator.reset()
        for step in range(generator.steps()):
            image, label, observation, masks, label_masks = generator.next_batch()
            dict = {
                input_images: image,
                input_characters: observation,
                sequence_masks: label_masks,
                image_masks: masks,
                y_tensor: label,
                lr: lr_val
            }
            if writer is not None and global_step % summary_step == 0:
                vloss, vacc, s, _ = sess.run([loss, accuracy, merged_summary, train], feed_dict=dict)
                writer.add_summary(s, global_step)
            else:
                vloss, vacc, _ = sess.run([loss, accuracy, train], feed_dict=dict)

            print("Loss: {}, Acc: {}".format(vloss, vacc))
            if (epoch + 1) % epoch_lr_decay == 0:
                lr_val = max(lr_val * decay_rate, 0.0001)
                lr_summary.value[0].simple_value = lr_val
                if writer is not None:
                    writer.add_summary(lr_summary, global_step)

            global_step += 1

        if level < levels - 1:
            level += 1
            generator.set_level(level)
        continue
        # Validation after each epoch
        wern = 0
        accn = 0
        exprate = 0
        for step in range(generator_valid.steps()):
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
            writer.add_summary(valid_avg_wer_summary, global_step)
            writer.add_summary(valid_avg_acc_summary, global_step)
            writer.add_summary(valid_avg_exp_rate_summary, global_step)
            writer.flush()

        improved = False

        #if avg_exp_rate > best_exp_rate:
        #    best_exp_rate = avg_exp_rate
        #    improved = True

        if avg_wer < best_wer:
            best_wer = avg_wer
            improved = True

        if improved:
            bad_counter = 0
            saver.save(sess, save_format.format(epoch))
        else:
            bad_counter += 1
        if bad_counter == patience:
            print("Early stopping")
            break

        generator_valid.reset()


