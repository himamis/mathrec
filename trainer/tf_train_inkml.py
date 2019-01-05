import file_utils as utils
from trainer import tf_model
from utilities import parse_arg
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
folder_str = folder_str + '-' + tensorboard_name
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

batch_size = 32
epochs = 1000
levels = 5
decay = 1e-4
encoding_vb, decoding_vb = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))

image, truth, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_train.pkl")))
image_valid, truth_valid, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_validate.pkl")))

generator = DataGenerator(image, truth, encoding_vb, batch_size=batch_size)
generator_valid = DataGenerator(image_valid, truth_valid, encoding_vb, 1)

if True:
    image, truth, _ = zip(*utils.read_pkl(path.join(data_base_dir, "overfit.pkl")))
    new_vocab = "1234567890-+"
    new_vocab = list(new_vocab)
    new_vocab.append("<end>")
    new_vocab.append("<start>")

    encoding_vb = dict(zip(new_vocab, range(len(new_vocab))))
    decoding_vb = { v: k for k, v in encoding_vb.items() if k != "<start>"}

    image_valid = image
    truth_valid = truth

    gen = simple_number_operation_generator()
    conf = Config()
    parser = Parser(create_graphics_factory(os.path.join(data_base_dir, 'tokengroup.pkl')))
    generator = TokenDataGenerator(gen, parser, conf, encoding_vb, batch_size, 10)
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
    eval_feature_grid, eval_masking = model.feature_grid(pl_input_images, pl_image_masks, is_training=pl_is_training,
                                                         summarize=False, r_max=pl_r_max, d_max=pl_d_max)
    eval_calculate_h0, eval_calculate_alphas = model.calculate_decoder_init(eval_feature_grid, eval_masking)
    output, (states_h, states_alpha) = model.decoder(eval_feature_grid, eval_masking, pl_input_characters,
                                                     eval_calculate_h0, eval_calculate_alphas, summarize=False)

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
pl_sequence_masks = tf.placeholder(dtype=t.my_tf_float, shape=(batch_size, None))

with tf.name_scope("loss"):
    #tf.summary.histogram("before_softmax", output)
    loss = tf.contrib.seq2seq.sequence_loss(output, pl_y_tensor, pl_sequence_masks)

    # L2 regularization
    #for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #    if not variable.name.startswith('batch_norm'):
    #        loss += decay * tf.reduce_sum(tf.pow(variable, 2))

    tf.summary.scalar("loss", loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    grads_and_vars = optimizer.compute_gradients(loss)
    # Gradient clipping
    # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    train = optimizer.apply_gradients(grads_and_vars)

#tf.summary.merge(
#    [tf.summary.histogram("gradient-{}".format(g[1].name), g[0]) for g in grads_and_vars if g[0] is not None])

with tf.name_scope("accuracy"):
    result = tf.argmax(tf.nn.softmax(output), output_type=tf.int32, axis=2)

    accuracy = tf.contrib.metrics.accuracy(result, pl_y_tensor, pl_sequence_masks)
    #accuracy = tf.Print(accuracy, [pl_is_training, result, pl_y_tensor, pl_sequence_masks],
    #                    "Training, Res, Tens, Maks", summarize=20)
    tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
no_summary_per_epoch = 2
summary_step = math.floor(generator.steps() / no_summary_per_epoch)
patience = 50
save_epoch = 50
bad_counter = 0
best_wer = 999999
best_exp_rate = -1
#level = 0
level = 4

r_max_val_init = 1
d_max_val_init = 0
r_max_val = r_max_val_init
d_max_val = d_max_val_init

valid_avg_wer_summary = tf.Summary()
valid_avg_acc_summary = tf.Summary()
valid_avg_exp_rate_summary = tf.Summary()
level_summary = tf.Summary()

valid_avg_wer_summary.value.add(tag="valid_avg_wer", simple_value=None)
valid_avg_acc_summary.value.add(tag="valid_avg_acc", simple_value=None)
valid_avg_exp_rate_summary.value.add(tag="valid_exp_rate", simple_value=None)
level_summary.value.add(tag="level", simple_value=None)

init = tf.global_variables_initializer()

print("Image2Latex Start training...")
global_step = 1

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
with tf.Session(config=config) as sess:
    if start_epoch != -1:
        saver.restore(sess, save_format.format(start_epoch))
        start_epoch = -1
    else:
        sess.run(init)
    predictor = create_predictor(sess,
                                 (pl_input_images, pl_image_masks, pl_is_training, pl_r_max, pl_d_max),
                                 (eval_feature_grid, eval_masking, eval_calculate_h0, eval_calculate_alphas),
                                 (pl_input_characters),
                                 (eval_output_softmax, states_h, states_alpha),
                                 encoding_vb, decoding_vb, k=10, max_length=10)
    writer = None
    if tensorboard_log_dir is not None:
        writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, tensorboard_name))
        writer.add_graph(sess.graph)

    for epoch in range(epochs):
        print("Staring epoch {}".format(epoch))
        #print("Current level {}".format(level))
        #level_summary.value[0].simple_value = level
        #if writer is not None:
        #    writer.add_summary(level_summary, global_step)

        generator.reset()
        for step in range(generator.steps()):
            break
            image, label, observation, masks, label_masks = generator.next_batch()
            dict = {
                pl_input_images: image,
                pl_input_characters: observation,
                pl_sequence_masks: label_masks,
                pl_image_masks: masks,
                pl_y_tensor: label,
                pl_is_training: True,
                pl_r_max: r_max_val,
                pl_d_max: d_max_val
            }
            if writer is not None and global_step % summary_step == 0:
                vloss, vacc, s, _ = sess.run([loss, accuracy, merged_summary, train], feed_dict=dict)
                writer.add_summary(s, global_step)
            else:
                vloss, vacc, _ = sess.run([loss, accuracy, train], feed_dict=dict)

            print("Loss: {}, Acc: {}".format(vloss, vacc))

            global_step += 1

            from_epoch = 250
            until_epoch = 350
            diff = max(min((epoch - from_epoch) / (until_epoch - from_epoch), 1), 0)
            r_max_val = r_max_val_init + 2 * diff
            d_max_val = d_max_val_init + 5 * diff
            print("Step {}: r_max_val {}, d_max_val {}".format(global_step, r_max_val, d_max_val))


        # if level < levels - 1:
        #     level += 1
        #     generator.set_level(level)

        # Validation after each epoch
        #if (epoch + 1) % 20 != 0:
        #    continue
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
            writer.add_summary(valid_avg_wer_summary, epoch)
            writer.add_summary(valid_avg_acc_summary, epoch)
            writer.add_summary(valid_avg_exp_rate_summary, epoch)
            writer.flush()

        improved = False

        #if avg_exp_rate > best_exp_rate:
        #    best_exp_rate = avg_exp_rate
        #    improved = True

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

