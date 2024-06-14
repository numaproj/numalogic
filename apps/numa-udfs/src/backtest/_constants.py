import os
from typing import Final

from numalogic._constants import BASE_DIR

DEFAULT_OUTPUT_DIR: Final[str] = os.path.join(BASE_DIR, ".btoutput")
DEFAULT_SEQUENCE_LEN: Final[int] = 12
DEFAULT_PROM_LOCALHOST: Final[str] = "http://localhost:9090/"
