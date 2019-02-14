from utilities import parse_arg


def _parse_boolean(name, required=False, default=False):
    boolean = parse_arg(name, default='t' if default else 'n', required=required)
    return False if 'n' == boolean else True


def _parse_int(name, required=False, default=None):
    integer = parse_arg(name, default=default, required=required)
    return int(integer) if integer is not None else None


gcs = parse_arg('--gcs', required=False)
use_gpu = parse_arg('--gpu', default='n', required=False)
start_epoch = int(parse_arg('--start-epoch', 0))
ckpt_dir = parse_arg('--ckpt-dir', None, required=False)
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/new_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/tf_model')
restore_weights_dir = parse_arg('--restore-weights', required=False, default=None)
tensorboard_log_dir = parse_arg('--tb', None, required=False)
tensorboard_name = parse_arg('--tbn', "adam", required=False)
git_hexsha = parse_arg('--git-hexsha', 'NAN')
profiling = parse_arg('--profiling', default='n', required=False)
data_format = parse_arg('--data-format', default='channels_last')
if use_gpu == 'n':
    assert data_format == 'channels_last', "Only channels_last data format is availabel with CPU"


verbose_summary = _parse_boolean('--verbose-summary', default=False)
verbose = _parse_boolean('--verbose', default=False)
use_new_rnn = _parse_boolean('--new-rnn', default=False)
allow_soft_placement = _parse_boolean('--allow-soft-placement', default=False)

# Spatial transformer parameters
use_spatial_transformer = _parse_boolean('--st', default=False)

if use_spatial_transformer:
    assert data_format == 'channels_last', 'Only channels_last data format is compatible with spatial transformers'

overfit = _parse_boolean('--overfit', default=False)
patience = int(parse_arg('--patience', default=15))
batch_size = int(parse_arg('--batch-size', default=32))
allow_growth = _parse_boolean('--allow-growth', default=False)
epochs = int(parse_arg('--epochs', default=500))
epoch_per_validation = int(parse_arg('--epv', default=2))
validate_on_training = _parse_boolean("--vot", default=False)

device = '/cpu:0' if use_gpu == 'n' else '/gpu:{}'.format(use_gpu)

head = _parse_int('--head', default=None)
hidden_size = _parse_int('--hidden-size', default=None)
layers = _parse_int('--hidden-layers', default=None)
beta = float(parse_arg('--beta', default=0))