import io
import logging
from copy import copy
from typing import Optional, Dict, Tuple, BinaryIO, Union, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import OutlierMixin
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from numalogic.tools.types import AutoencoderModel

_LOGGER = logging.getLogger(__name__)


class AutoencoderPipeline(OutlierMixin):
    r"""
    Class to simplify training, inference, loading and saving of time-series autoencoders.

    Note:
         This class only supports Pytorch models.
    Args:
        model: model instance
        seq_len: sequence length
        loss_fn: loss function used for training
                        supported values include {"huber", "l1", "mse"}
        optimizer: optimizer to used for training.
                           supported values include {"adam", "adagrad", "rmsprop"}
        lr: learning rate
        batch_size: batch size for training
        num_epochs: number of epochs for training
        std_tolerance: determines how many times the standard deviation to be used for threshold
        reconerr_method: method used to calculate the distance
                                between the original and the reconstucted data
                                supported values include {"absolute", "squared"}
        threshold_min: the minimum threshold to use;
                              can be used when the threshold calculated is too low

    >>> # Example usage
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> x = np.random.randn(100, 3)
    >>> seq_len = 10
    >>> model = VanillaAE(signal_len=seq_len, n_features=3)
    >>> ae_trainer = AutoencoderPipeline(model=model, seq_len=seq_len)
    >>> ae_trainer.fit(x)
    """

    def __init__(
        self,
        model: AutoencoderModel = None,
        seq_len: int = None,
        loss_fn: str = "huber",
        optimizer: str = "adam",
        lr: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 100,
        std_tolerance: float = 3.0,
        reconerr_method: str = "absolute",
        threshold_min: float = None,
    ):
        if not (model and seq_len):
            raise ValueError("No model and seq len provided!")

        self._model = model
        self.seq_len = seq_len
        self.criterion = self.init_criterion(loss_fn)
        self.optimizer = self.init_optimizer(optimizer, lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self._thresholds = None
        self._stats: Dict[str, Optional[float]] = dict(mean=None, std=None)
        self.stdtol = std_tolerance
        self.reconerr_func = self.get_reconerr_func(reconerr_method)
        self.threshold_min = threshold_min

    @property
    def model_properties(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "thresholds": self._thresholds,
            "err_stats": self._stats,
        }

    @property
    def model(self) -> AutoencoderModel:
        return self._model

    @property
    def thresholds(self) -> Optional[NDArray[float]]:
        return self._thresholds

    @property
    def err_stats(self) -> Dict[str, Optional[NDArray[float]]]:
        return self._stats

    @staticmethod
    def get_reconerr_func(method: str) -> Callable:
        if method == "squared":
            return np.square
        if method == "absolute":
            return np.abs
        raise ValueError(f"Unrecognized reconstruction error method specified: {method}")

    @staticmethod
    def init_criterion(loss_fn: str):
        if loss_fn == "huber":
            return nn.HuberLoss(delta=0.5)
        if loss_fn == "l1":
            return nn.L1Loss()
        if loss_fn == "mse":
            return nn.MSELoss()
        raise NotImplementedError(f"Unsupported loss function provided: {loss_fn}")

    def init_optimizer(self, optimizer: str, lr: float):
        if optimizer == "adam":
            return optim.Adam(self._model.parameters(), lr=lr)
        if optimizer == "adagrad":
            return optim.Adagrad(self._model.parameters(), lr=lr)
        if optimizer == "rmsprop":
            return optim.RMSprop(self._model.parameters(), lr=lr)
        raise NotImplementedError(f"Unsupported optimizer value provided: {optimizer}")

    def fit(self, X: NDArray[float], y=None, log_freq: int = 5) -> "AutoencoderPipeline":
        r"""
        Fit function to train autoencoder model

        Args:
            X: training dataset
            y: labels
            log_freq: frequency logging

        Returns:
            AutoencoderPipeline instance
        """
        _LOGGER.info("Training autoencoder model..")

        dataset = self._model.construct_dataset(X, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._model.train()

        loss = torch.Tensor([0.0])
        for epoch in range(1, self.num_epochs + 1):
            for x_batch in loader:
                self.optimizer.zero_grad()
                _, decoded = self._model(x_batch)
                loss = self.criterion(decoded, x_batch)
                loss.backward()
                self.optimizer.step()

            if epoch % log_freq == 0:
                _LOGGER.info(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")

        self._thresholds, _mean, _std = self.find_thresholds(X)
        self._stats["mean"] = _mean
        self._stats["std"] = _std

        return self

    def predict(self, X: NDArray[float], seq_len: int = None) -> NDArray[float]:
        r"""
        Return the reconstruction from the model.

        Args:
            X: training dataset
            seq_len: sequence length / window length

        Returns:
            Numpy array
        """
        if not seq_len:
            seq_len = self.seq_len or len(X)
        dataset = self._model.construct_dataset(X, seq_len)
        self._model.eval()
        with torch.no_grad():
            _, pred = self._model(dataset.data)
        return dataset.recover_shape(pred)

    def score(self, X: NDArray[float], seq_len: int = None) -> NDArray[float]:
        r"""
        Return anomaly score using the calculated threshold

        Args:
            X: training dataset
            seq_len: sequence length / window length

        Returns:
            numpy array with anomaly scores
        """
        if self._thresholds is None:
            raise RuntimeError("Thresholds not present!!!")
        thresh = self._thresholds.reshape(1, -1)
        if not seq_len:
            seq_len = self.seq_len or len(X)
        recon_err = self.recon_err(X, seq_len=seq_len)
        anomaly_scores = recon_err / thresh
        return anomaly_scores

    def recon_err(self, X: NDArray[float], seq_len: int) -> NDArray:
        r"""
        Returns the reconstruction error.

        Args:
            X: training dataset
            seq_len: sequence length / window length

        Returns:
            numpy array with anomaly scores
        """
        x_recon = self.predict(X, seq_len=seq_len)
        recon_err = self.reconerr_func(X - x_recon)
        return recon_err

    def find_thresholds(
        self, X: NDArray[float]
    ) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        r"""
        Calculate threshold for the anomaly model
        Args:
            X: training dataset

        Returns:
            Tuple consisting of thresholds, reconstruction error mean, reconstruction error std
        """
        recon_err = self.recon_err(X, seq_len=self.seq_len)
        recon_err_mean = np.mean(recon_err, axis=0)
        recon_err_std = np.std(recon_err, axis=0)
        thresholds = recon_err_mean + (self.stdtol * recon_err_std)
        if self.threshold_min:
            thresholds[thresholds < self.threshold_min] = self.threshold_min
        return thresholds, recon_err_mean, recon_err_std

    def save(self, path: Optional[str] = None) -> Optional[BinaryIO]:
        r"""
        Save function to save the model.
        If path is provided then the model is saved in the given path.

        Args:
              path: path to save the model (Optional parameter)
        Returns:
              Binary type object if path is None
        """
        state_dict = copy(self.model_properties)
        state_dict["model_state_dict"] = self._model.state_dict()
        if path:
            torch.save(state_dict, path)
        else:
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            return buf

    def __load_metadata(self, **metadata) -> None:
        self.optimizer.load_state_dict(metadata["optimizer_state_dict"])
        self._thresholds = metadata["thresholds"]
        self._stats = metadata["err_stats"]

    def load(self, path: Union[str, BinaryIO] = None, model=None, **metadata) -> None:
        r"""
        Load the model to pipeline.

        Args:
              path: path to load the model
              model: machine learning model
              metadata: additional pipeline metadata
        """
        if (path and model) or (not path and not model):
            raise ValueError("One of path or model needs to be provided!")
        if model:
            self._model = model
            if metadata:
                self.__load_metadata(**metadata)
            return
        if path:
            if isinstance(path, io.BytesIO):
                path.seek(0)
                checkpoint = torch.load(path)
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self.__load_metadata(**checkpoint)
            elif isinstance(path, str):
                checkpoint = torch.load(path)
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self.__load_metadata(**checkpoint)
            return

    @classmethod
    def with_model(
        cls,
        model_cls,
        seq_len: int,
        loss_fn="huber",
        optimizer="adam",
        lr=0.001,
        batch_size=256,
        num_epochs=100,
        **model_kw,
    ) -> "AutoencoderPipeline":
        model = model_cls(**model_kw)
        return AutoencoderPipeline(
            model,
            seq_len,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )


class SparseAEPipeline(AutoencoderPipeline):
    def __init__(self, beta=1e-3, rho=0.05, method="kl_div", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.rho = rho
        self.reg_method = method

    def l1_loss(self) -> Tensor:
        l1_lambda = self.beta
        l1_norm = sum(torch.linalg.norm(p, 1) for p in self._model.parameters())
        return torch.Tensor(l1_norm + l1_lambda)

    def l2_loss(self) -> Tensor:
        l2_lambda = self.beta
        l2_norm = sum(torch.linalg.norm(p, 2) for p in self._model.parameters())
        return torch.Tensor(l2_norm + l2_lambda)

    def kl_divergence(self, activations: Tensor) -> Tensor:
        rho_hat = torch.mean(activations, dim=0)
        rho = torch.full(rho_hat.size(), self.rho)
        kl_loss = nn.KLDivLoss(reduction="sum")
        _dim = 0 if rho_hat.dim() == 1 else 1
        return kl_loss(torch.log_softmax(rho_hat, dim=_dim), torch.softmax(rho, dim=_dim))

    def calculate_regularized_loss(self, activation: Tensor) -> Tensor:
        if self.reg_method == "kl_div":
            return self.kl_divergence(activation) * self.beta
        if self.reg_method == "L1":
            return self.l1_loss()
        if self.reg_method == "L2":
            return self.l2_loss()
        raise NotImplementedError(
            f"Unsupported regularization method value provided: {self.reg_method}"
        )

    def fit(self, X: NDArray[float], y=None, log_freq: int = 5) -> None:
        _LOGGER.info(
            "Training sparse autoencoder model with beta: %s, and rho: %s", self.beta, self.rho
        )
        _LOGGER.info("Using %s regularized loss", self.reg_method)
        dataset = self._model.construct_dataset(X, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._model.train()

        loss, penalty = torch.Tensor([0.0]), torch.Tensor([0.0])
        for epoch in range(1, self.num_epochs + 1):
            for x_batch in loader:
                self.optimizer.zero_grad()
                encoded, decoded = self._model(x_batch)

                loss = self.criterion(decoded, x_batch)
                penalty = self.calculate_regularized_loss(encoded)
                loss += penalty
                loss.backward()
                self.optimizer.step()

            if epoch % log_freq == 0:
                _LOGGER.info(f"epoch : {epoch}, penalty: {penalty} loss_mean : {loss.item():.7f}")

        self._thresholds, _mean, _std = self.find_thresholds(X)
        self._stats["mean"] = _mean
        self._stats["std"] = _std
