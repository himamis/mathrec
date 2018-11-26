import sys

from .common import *
if '--gcs' in sys.argv:
    print('Image2Latex: use google cloud storage')
    from .gcs_file_utils import *
    import google.cloud.logging

    # Instantiates a client
    client = google.cloud.logging.Client()

    # Connects the logger to the root logging handler; by default this captures
    # all logs at INFO level and higher
    client.setup_logging()
else:
    print('Image2Latex: use local storage')
    from .local_file_utils import *
