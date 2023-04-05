import sys
import os
import warnings

if not sys.warnoptions:
    warnings.simplefilter("default", category=UserWarning)
    os.environ["PYTHONWARNINGS"] = "default"
