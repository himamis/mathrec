import sys


if '--gcs' in sys.argv:
    print('Image2Latex: use google cloud storage')
    from . import gcs_file_utils as utils
else:
    print('Image2Latex: use local storage')
    from . import local_file_utils as utils
