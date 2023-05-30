import logging
import sys
import os
import warnings

if not sys.warnoptions:
    warnings.simplefilter("default", category=UserWarning)
    os.environ["PYTHONWARNINGS"] = "default"


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
