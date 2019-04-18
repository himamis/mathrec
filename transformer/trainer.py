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
from utils import metrics
import time

random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)


def log(message):
    if params.verbose:
        print(message)


def create_generators(batch_size=32):
    if not params.validate_only:
        training = read_pkl(path.join(params.data_base_dir, params.training_fname + ".pkl"))
        training_generator = generator.DataGenerator(training, batch_size, do_shuffle=True)
    else:
        training_generator = None

    validation_data = "validation.pkl"
    if params.validate_on_training:
        validation_data = params.training_fname + ".pkl"
    validating = read_pkl(path.join(params.data_base_dir, validation_data))
    # validating_batch_size = int(batch_size / 2)
    validating_batch_size = batch_size
    validating_generator = generator.DataGenerator(validating, validating_batch_size, do_shuffle=False)

    return training_generator, validating_generator

# def create_generators(batch_size=32):
#     training = read_pkl(path.join(params.data_base_dir, 'training_data.pkl'))
#     training_generator = generator.DataGenerator(training, batch_size, do_shuffle=True, steps=0)
#
#     validation_data = "validating_data.pkl"
#     if params.validate_on_training:
#         validation_data = "training_data.pkl"
#     vv = read_pkl(path.join(params.data_base_dir, validation_data))
#     validating = []
#     for (formula, input) in vv:
#         form_string = "".join(formula)
#         if "A+A+B+B+C" in form_string or "x\\timesx\\timesx" in form_string or \
#                 "1+1+1+" in form_string or \
#                 "9999" in form_string or \
#                 "A+A+B+B+C" in form_string or \
#                 "100000000" in form_string:
#             validating.append((formula, input))
#
#     validating_batch_size = 1
#     validating_generator = generator.DataGenerator(validating, validating_batch_size, do_shuffle=False)
#
#     return training_generator, validating_generator



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


def create_tex_file(output_path, name, latex):
    filename = path.join(output_path, name + ".txt")
    tex_file = open(filename, 'w')
    tex_file.write(latex)
    tex_file.close()


def validate(sess, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder, validating):
    wern = 0
    exprate = 0
    accn = 0
    no = 0

    inputs = []
    wers = []
    results = []
    targets = []
    for validation_step in range(validating.steps()):
        progress_bar("Validating", validation_step + 1, validating.steps())
        name, encoded_tokens, bounding_boxes, encoded_formulas, _ = validating.next_batch()
        feed_dict = {
            tokens_placeholder: encoded_tokens,
            bounding_box_placeholder: bounding_boxes,
            output_placeholder: encoded_formulas  # ,
            # output_masks_placeholder: encoded_formulas_masks
        }
        outputs = sess.run(eval_fn, feed_dict)

        for i in range(len(outputs['outputs'])):
            result = outputs['outputs'][i]
            result = np.trim_zeros(result, 'b')  # Remove padding zeroes from the end
            target = encoded_formulas[i]
            target = np.trim_zeros(target, 'b')  # Remove padding zeroes from the end
            log("Validation: \n Expected: \t {}\nResult: \t {}".format(target, result))
            cwer = wer(result, target) / max(len(target), len(result))
            wern += cwer
            exprate += exp_rate(target, result)
            if abs(cwer) < 1e-6:
                accn += 1
            no += 1

            inputs.append(encoded_tokens[i])
            results.append(result)
            targets.append(target)
            wers.append(cwer)

            if params.create_tex_files is not None:
                # Remove <END> symbol
                result = result[:-1]
                result = vocabulary.decode_formula(result)
                create_tex_file(params.create_tex_files, name[i], result)

    validating.reset()

    avg_wer = float(wern) / float(no)
    avg_acc = float(accn) / float(no)
    avg_exp_rate = float(exprate) / float(no)

    # Print expressions that perform bad
    print("Underperforming formulas (avg weg {}):\n".format(avg_wer))
    for index, cwer in enumerate(wers):
        if cwer > avg_wer:
            result = results[index]
            target = targets[index]
            input = vocabulary.decode_formula(inputs[index], join=False)
            decoded_result = vocabulary.decode_formula(result)
            decoded_target = vocabulary.decode_formula(target)
            print("Wer: {}\nInput: {}\nTarget: {}\nResult: {}\n\n".format(cwer, input, decoded_target, decoded_result))

    return avg_wer, avg_acc, avg_exp_rate


def create_config():
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=params.allow_soft_placement)
    config.gpu_options.allow_growth = params.allow_growth
    return config


def create_saver_and_save_path():
    saver = tf.train.Saver(name=params.tensorboard_name, pad_step_number=True)
    save_path = path.join(params.model_checkpoint_dir, params.tensorboard_name)

    return saver, save_path


def restore_model(sess, saver, save_path, epoch):
    saver.restore(sess, save_path + "-%08d" % epoch)


def train_loop(sess, train, eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder,
               output_masks_placeholder):
    training, validating = create_generators(params.batch_size)
    saver, save_path = create_saver_and_save_path()

    tf_epoch = tf.Variable(0, dtype=tf.int64, name="epoch")
    with tf.name_scope("evaluation"):
        tf_avg_wer = tf.Variable(1, dtype=tf.float32, name="avg_wer")
        tf_avg_acc = tf.Variable(0, dtype=tf.float32, name="avg_acc")
        tf_avg_exp_rate = tf.Variable(0, dtype=tf.float32, name="avg_exp_rate")

        with tf.contrib.summary.record_summaries_every_n_global_steps(params.epoch_per_validation, tf_epoch):
            tf.contrib.summary.scalar("avg_wer", tf_avg_wer, "validation", tf_epoch)
            tf.contrib.summary.scalar("avg_acc", tf_avg_acc, "validation", tf_epoch)
            tf.contrib.summary.scalar("avg_exp_rate", tf_avg_exp_rate, "validation", tf_epoch)

    tf.global_variables_initializer().run()

    if params.start_epoch != 0:
        restore_model(sess, saver, save_path, params.start_epoch)

    tf.contrib.summary.initialize(graph=tf.get_default_graph())
    best_accuracy = -1

    for epoch in range(params.start_epoch, params.epochs):
        sess.run(tf.assign(tf_epoch, epoch))
        steps = training.steps()
        for step in range(steps):
            progress_bar("Epoch {}".format(epoch + 1), step + 1, steps)
            _, encoded_tokens, bounding_boxes, encoded_formulas, encoded_formulas_masks = training.next_batch()
            feed_dict = {
                tokens_placeholder: encoded_tokens,
                bounding_box_placeholder: bounding_boxes,
                output_placeholder: encoded_formulas,
                output_masks_placeholder: encoded_formulas_masks
            }
            sess.run([train, tf.contrib.summary.all_summary_ops()], feed_dict)

        # writer.flush()
        tf.contrib.summary.flush()
        training.reset()

        if (epoch + 1) % params.epoch_per_validation != 0:
            continue

        avg_wer, avg_acc, avg_exp_rate = validate(sess, eval_fn, tokens_placeholder, bounding_box_placeholder,
                                                  output_placeholder, validating)
        sess.run([
            tf.assign(tf_avg_wer, avg_wer),
            tf.assign(tf_avg_exp_rate, avg_exp_rate),
            tf.assign(tf_avg_acc, avg_acc)
        ])

        if best_accuracy < avg_acc:
            best_accuracy = avg_acc
            saver.save(sess, save_path, epoch + 1)


def update_params(transformer_params):
    encoding_vb = vocabulary.encoding_vocabulary
    transformer_params.update(vocab_size=len(encoding_vb))
    if params.head is not None:
        transformer_params.update(num_heads=params.head)
    if params.layers is not None:
        transformer_params.update(num_hidden_layers=params.layers)
    if params.hidden_size is not None:
        transformer_params.update(hidden_size=params.hidden_size)
    if params.filter_size is not None:
        transformer_params.update(filter_size=params.filter_size)
    transformer_params.update(beta=params.beta)
    transformer_params.update(alpha=params.alpha)
    transformer_params.update(beam_size=params.beam_size)


def create_eval_train_fns(transformer_params, tokens_placeholder, bounding_box_placeholder,
                          output_placeholder, output_masks_placeholder):
    with tf.device(params.device):
        model = create_model(transformer_params)
        logits = model(tokens_placeholder, bounding_box_placeholder, output_placeholder, True)

        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, output_placeholder,
            transformer_params["label_smoothing"],
            transformer_params["vocab_size"])
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        # Create loss function
        # loss = tf.contrib.seq2seq.sequence_loss(logits, output_placeholder, output_masks_placeholder)
        # L2 regularization
        decay = transformer_params["l2_regularization"]
        if decay is not None:
            for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                loss += decay * tf.reduce_sum(tf.pow(variable, 2))

        # Create Optimizer
        learning_rate = get_learning_rate(
            learning_rate=transformer_params["learning_rate"],
            hidden_size=transformer_params["hidden_size"],
            learning_rate_warmup_steps=transformer_params["learning_rate_warmup_steps"])
        # learning_rate = tf.to_float(transformer_params["learning_rate"])

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
        # tf.contrib.summary.image("diffs",
        #                         tf.get_default_graph().get_tensor_by_name(
        #                             "transformer/Transformer/encode/create_diffs/diffs:0"))

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

    return train, eval_fn, writer


def train(transformer_params, tokens_placeholder, bounding_box_placeholder,
          output_placeholder, output_masks_placeholder):
    train_fn, eval_fn, writer = create_eval_train_fns(transformer_params, tokens_placeholder, bounding_box_placeholder,
                          output_placeholder, output_masks_placeholder)
    with writer.as_default():
        with tf.Session(config=create_config()) as sess:
            train_loop(sess, train_fn, eval_fn, tokens_placeholder, bounding_box_placeholder,
                       output_placeholder, output_masks_placeholder)


def test(transformer_params, tokens_placeholder, bounding_box_placeholder,
         output_placeholder, output_masks_placeholder):
    batch_size = params.batch_size
    if params.create_tex_files is not None:
        # Create results one by one
        # As I am not sure, if the padding
        # Influences other predictions
        # This makes inference slower
        batch_size = 1
    _, validating = create_generators(batch_size )
    with tf.Session(config=create_config()) as sess:
        _, eval_fn, _ = create_eval_train_fns(transformer_params, tokens_placeholder, bounding_box_placeholder,
                                                  output_placeholder, output_masks_placeholder)
        saver, save_path = create_saver_and_save_path()
        restore_model(sess, saver, save_path, params.start_epoch)
        avg_wer, avg_acc, avg_exp_rate = validate(sess, eval_fn, tokens_placeholder, bounding_box_placeholder,
                                                  output_placeholder, validating)
        print("Done validating\n Avg_wer: {}\nAvg_acc: {}\nAvg_exp_rate: {}\n".format(avg_wer, avg_acc,
                                                                                      avg_exp_rate))


def main(transformer_params):
    update_params(transformer_params)

    tokens_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="tokens")
    bounding_box_placeholder = tf.placeholder(tf.float32, shape=(None, None, 4), name="bounding_boxes")

    output_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="output")
    output_masks_placeholder = tf.placeholder(tf.float32, shape=(None, None), name="output_masks")

    if params.validate_only:
        test(transformer_params, tokens_placeholder, bounding_box_placeholder,
             output_placeholder, output_masks_placeholder)
    else:
        train(transformer_params, tokens_placeholder, bounding_box_placeholder,
              output_placeholder, output_masks_placeholder)


def prepare_transform():
    transformer_params = model_params.CUSTOM_PARAMS
    update_params(transformer_params)

    tokens_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="tokens")
    bounding_box_placeholder = tf.placeholder(tf.float32, shape=(None, None, 4), name="bounding_boxes")

    output_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="output")
    output_masks_placeholder = tf.placeholder(tf.float32, shape=(None, None), name="output_masks")
    session = tf.Session(config=create_config())
    with session.as_default():
        _, eval_fn, _ = create_eval_train_fns(transformer_params, tokens_placeholder, bounding_box_placeholder,
                                              output_placeholder, output_masks_placeholder)
        saver, save_path = create_saver_and_save_path()
        restore_model(session, saver, save_path, params.start_epoch)

    return (eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder, session)


def normalize(inputs):
    info = np.finfo(np.float32)

    min_x = info.max
    min_y = info.max
    max_x = info.min
    max_y = info.min

    for token, bounding_box in inputs:
        min_x = min(min_x, bounding_box[0])
        min_y = min(min_y, bounding_box[1])
        max_x = max(max_x, bounding_box[2])
        max_y = max(max_y, bounding_box[3])

    trans_x = min_x
    trans_y = min_y
    scale_x = 1 / (max_x - min_x)
    scale_y = 1 / (max_y - min_y)
    result = []
    for token, bounding_box in inputs:
        box = ((bounding_box[0] - trans_x) * scale_x,
               (bounding_box[1] - trans_y) * scale_y,
               (bounding_box[2] - trans_x) * scale_x,
               (bounding_box[3] - trans_y) * scale_y)
        if type(token) is str:
            token = vocabulary.encoding_vocabulary[token]
        result.append((token, box))

    sorted_result = sorted(result, key=lambda inp: inp[1][0])

    sorted_result.append((vocabulary.EOS_ID, (1.0, 1.0, 1.0, 1.0)))

    return sorted_result


def transform(model, inputs):
    try:
        inputs = normalize(inputs)
    except KeyError:
        time.sleep(1)
        return ""
    tokens, bounding_boxes = zip(*inputs)
    eval_fn, tokens_placeholder, bounding_box_placeholder, output_placeholder, session = model
    with session.as_default():
        feed_dict = {
            tokens_placeholder: [tokens],
            bounding_box_placeholder: [bounding_boxes]
        }
        outputs = session.run(eval_fn, feed_dict)
        result = outputs['outputs'][0]
    return vocabulary.decode_formula(result, join=True)
