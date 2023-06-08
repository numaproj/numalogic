import os

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(BASE_DIR)[0]
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
TESTS_RESOURCES = os.path.join(TESTS_DIR, "resources")
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# UDF constants
TRAIN_VTX_KEY = "train"
INFERENCE_VTX_KEY = "inference"
THRESHOLD_VTX_KEY = "threshold"
POSTPROC_VTX_KEY = "postproc"

CONFIG_PATHS = ["./config/user-configs", "./config/default-configs"]
