import sys


if '--gcs' in sys.argv:
    print('Image2Latex: use google cloud storage')
    from .gcs_file_utils import *
else:
    print('Image2Latex: use local storage')
    from .local_file_utils import *
