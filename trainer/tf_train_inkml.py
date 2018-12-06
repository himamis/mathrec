import file_utils as utils
from trainer import tf_model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path
import os
import logging
from trainer.tf_generator import DataGenerator
from trainer.tf_predictor import create_predictor
from trainer.metrics import wer
import trainer.default_type as t
import trainer.tf_initializers as tfi

logging.basicConfig(level=logging.DEBUG)

from tensorflow import set_random_seed
import tensorflow as tf

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
weights_fname = 'weights_{epoch}.h5'
history_fname = 'history.pkl'
results_fname = 'results.pkl'

gcs = parse_arg('--gcs', required=False)
use_gpu = parse_arg('--gpu', default='n', required=False)
start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/new_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/tf_model')
tensorboard_log_dir = parse_arg('--tb', None, required=False)
tensorboard_name = parse_arg('--tbn', "adam", required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
save_dir = base_dir
if gcs is not None:
    save_dir = os.path.join("gs://{}".format(gcs), save_dir)

#if not path.exists(base_dir):
#    os.mkdir(base_dir)

model_weights_file = path.join(model_checkpoint_dir, folder_str, weights_fname)
results_file = path.join(model_checkpoint_dir, folder_str, results_fname)
history_file = path.join(model_checkpoint_dir, folder_str, history_fname)

start_time = datetime.now()
git_hexsha = parse_arg('--git-hexsha', 'NAN')

batch_size = 8
epochs = 50
encoding_vb, decoding_vb = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))

image, truth, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_train.pkl")))
generator = DataGenerator(image, truth, encoding_vb, batch_size)

image_valid, truth_valid, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_validate.pkl")))
generator_valid = DataGenerator(image_valid, truth_valid, encoding_vb, 1)

input_images = tf.placeholder(t.my_tf_float, shape=(batch_size, None, None, 1), name="input_images")
image_masks = tf.placeholder(t.my_tf_float, shape=(batch_size, None, None, 1), name="input_image_masks")
input_characters = tf.placeholder(tf.int32, shape=(batch_size, None), name="input_characters")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

single_image = tf.placeholder(t.my_tf_float, shape=(1, None, None, 1), name="single_input_image")
single_image_mask = tf.placeholder(t.my_tf_float, shape=(1, None, None, 1), name="single_input_image_mask")
single_character = tf.placeholder(tf.int32, shape=(1, 1), name="single_character")

logging.debug("Image2Latex: Start create model: {}".format(str(datetime.now().time())))
device = '/cpu:0' if use_gpu == 'n' else '/gpu:{}'.format(use_gpu)
with tf.device(device):
    model = tf_model.Model(len(encoding_vb),
                           filter_sizes=[32, 64, 128],
                           encoder_size=32,
                           decoder_units=64,
                           attention_dim=64,
                           embedding_dim=64,
                           bidirectional=True,
                           conv_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           conv_bias_init=tf.initializers.constant(0.01, dtype=t.my_tf_float),
                           conv_activation=tf.nn.relu,
                           encoder_kernel_init=tf.initializers.random_normal(dtype=t.my_tf_float),
                           decoder_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           decoder_bias_init=tf.initializers.constant(1/4, dtype=t.my_tf_float),
                           dense_init=tf.initializers.random_normal(stddev=0.1, dtype=t.my_tf_float),
                           dense_bias_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float),
                           decoder_recurrent_kernel_init=tf.contrib.layers.xavier_initializer(dtype=t.my_tf_float))
    # Training
    training_output = model.training(input_images, image_masks, input_characters)

    # Evaluating
    eval_feature_grid, eval_masking = model.feature_grid(single_image, single_image_mask, False)
    eval_calculate_h0, eval_calculate_c0 = model.calculate_decoder_init(eval_feature_grid, eval_masking)
    eval_init_h, eval_init_c = model.decoder_init(1)
    eval_state_h, eval_state_c, eval_output = model.decoder(eval_feature_grid, eval_masking,
                                                            single_character, eval_init_h, eval_init_c)
    eval_output_softmax = tf.nn.softmax(eval_output)

logging.debug("Image2Latex: End create model: {}".format(str(datetime.now().time())))

y_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size, None), name="y_labels")
lengts_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name="lengths")


with tf.name_scope("loss"):
    sequence_masks = tf.sequence_mask(lengts_tensor, dtype=t.my_tf_float)
    tf.summary.histogram("before_softmax,", training_output)
    loss = tf.contrib.seq2seq.sequence_loss(training_output, y_tensor, sequence_masks)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
    train = optimizer.minimize(loss)

with tf.name_scope("accuracy"):
    result = tf.argmax(tf.nn.softmax(training_output), output_type=tf.int32, axis=2)
    accuracy = tf.contrib.metrics.accuracy(result, y_tensor, sequence_masks)
    tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
summary_step = 10
patience = 4
bad_counter = 0
best_wer = 999999

valid_avg_wer_summary = tf.Summary()
valid_avg_acc_summary = tf.Summary()

init = tf.global_variables_initializer()

logging.debug("Image2Latex Start training...")
global_step = 0
with tf.Session() as sess:
    if start_epoch != 0:
        saver.restore(sess, save_dir)
    predictor = create_predictor(sess, (single_image, single_image_mask, eval_init_h,
                                 eval_init_c, eval_feature_grid, eval_masking,
                                 single_character), (eval_feature_grid, eval_masking, eval_calculate_h0,
                                                     eval_calculate_c0, eval_output_softmax, eval_state_h,
                                                     eval_state_c), encoding_vb, decoding_vb, k=10)
    writer = None
    if tensorboard_log_dir is not None:
        writer = tf.summary.FileWriter(os.path.join(tensorboard_log_dir, tensorboard_name))
        writer.add_graph(sess.graph)
    sess.run(init)
    for epoch in range(epochs):
        generator.reset()
        for step in range(generator.steps()):
            image, label, observation, masks, lengths = generator.next_batch()
            dict = {
                input_images: image,
                input_characters: observation,
                lengts_tensor: lengths,
                image_masks: masks,
                y_tensor: label
            }
            # Write summary
            if writer is not None and global_step % 10 == 0:
                s = sess.run(merged_summary, dict)
                writer.add_summary(s, global_step)

            vloss, vacc, _ = sess.run([loss, accuracy, train], feed_dict=dict)
            logging.debug("Loss: {}, Acc: {}".format(vloss, vacc))

            global_step += 1

        # Validation after each epoch
        wern = 0
        accn = 0
        for step in range(generator_valid.steps()):
            image, label, observation, masks, lengths = generator_valid.next_batch()
            predict = predictor(image, masks)
            re_encoded = [encoding_vb[s] for s in predict]
            wern += wer(re_encoded, label[0][:-1])
            if re_encoded == predict:
                accn += 1

        avg_wer = wern / generator_valid.steps()
        avg_acc = accn / generator_valid.steps()

        logging.debug("Avg_wer: {}, avg_acc: {}".format(avg_wer, avg_acc))

        valid_avg_wer_summary.value.add(tag="valid_avg_wer", simple_value=avg_wer)
        valid_avg_acc_summary.value.add(tag="valid_avg_acc", simple_value=avg_acc)
        if writer is not None:
            writer.add_summary(valid_avg_wer_summary, global_step)
            writer.add_summary(valid_avg_acc_summary, global_step)
            writer.flush()

        if avg_wer < best_wer:
            best_wer = avg_wer
            bad_counter = 0
            saver.save(sess, save_dir)
        else:
            bad_counter += 1
        if bad_counter == patience:
            logging.debug("Early stopping")
            break

        generator_valid.reset()


