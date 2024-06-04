# TODO move this subpackage to numalogic core library

from importlib.util import find_spec
from src.backtest._prom import PromBacktester, OutDataFrames


def _validate_req_pkgs():
    if (not find_spec("torch")) or (not find_spec("pytorch_lightning")):
        raise ModuleNotFoundError(
            "Pytorch and/or Pytorch lightning is not installed. Please install them first."
        )


_validate_req_pkgs()


__all__ = ["PromBacktester", "OutDataFrames"]
