from utilities import parse_arg


def _parse_boolean(name, required=False, default=False):
    boolean = parse_arg(name, default='t' if default else 'n', required=required)
    return False if 'n' == boolean else True


gcs = parse_arg('--gcs', required=False)
use_gpu = parse_arg('--gpu', default='n', required=False)
start_epoch = int(parse_arg('--start-epoch', -1))
ckpt_dir = parse_arg('--ckpt-dir', None, required=False)
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/new_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/tf_model')
tensorboard_log_dir = parse_arg('--tb', None, required=False)
tensorboard_name = parse_arg('--tbn', "adam", required=False)
git_hexsha = parse_arg('--git-hexsha', 'NAN')
profiling = parse_arg('--profiling', default='n', required=False)
data_format = parse_arg('--data-format', default='channels_last')
if use_gpu == 'n':
    data_format = 'channels_last'
    print('Changing data_format because cpu is used')


verbose_summary = _parse_boolean('--verbose-summary', default=False)
use_new_rnn = _parse_boolean('--new-rnn', default=False)
allow_soft_placement = _parse_boolean('--allow-soft-placement', default=False)
