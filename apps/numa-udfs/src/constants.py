import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BASE_DIR, "../tests")

BASE_CONF_DIR = os.path.join(BASE_DIR, "config")

DEFAULT_BASE_CONF_PATH = os.path.join(BASE_CONF_DIR, "default-configs", "config.yaml")
DEFAULT_APP_CONF_PATH = os.path.join(BASE_CONF_DIR, "app-configs", "config.yaml")
DEFAULT_METRICS_PORT = 8490
NUMALOGIC_METRICS = "numalogic_metrics"
