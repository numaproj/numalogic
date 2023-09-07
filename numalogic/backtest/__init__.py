try:
    import torch
    import pytorch_lightning
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Pytorch and/or Pytorch lightning is not installed. Please install it first."
    ) from None
