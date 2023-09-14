from importlib.util import find_spec


def _validate_req_pkgs():
    if (not find_spec("torch")) or (not find_spec("pytorch_lightning")):
        raise ModuleNotFoundError(
            "Pytorch and/or Pytorch lightning is not installed. Please install them first."
        )


_validate_req_pkgs()
