import sys

if '--gcs' in sys.argv:
    print('Image2Latex: use google cloud storage')
    import trainer.gcs_file_utils as utils
else:
    print('Image2Latex: use local storage')
    import trainer.local_file_utils as utils

from trainer.keras_implementations import AttentionDecoderLSTMCell, ModelCheckpointer, SequenceGenerator, RNN, Reshape, AttentionLSTMCell