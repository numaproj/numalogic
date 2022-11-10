import io
import logging
from copy import copy
from typing import Optional, BinaryIO, Union

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import TransformerMixin, BaseEstimator
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from numalogic.tools.types import AutoencoderModel

_LOGGER = logging.getLogger(__name__)


class AutoencoderPipeline(TransformerMixin, BaseEstimator):
    r"""
    Class to simplify training, inference, loading and saving of time-series autoencoders.

    Note:
         This class only supports Pytorch models.
    Args:
        model: model instance
        seq_len: sequence length
        loss_fn: loss function used for training
                        supported values include    {"huber", "l1", "mse"}
        optimizer: optimizer to used for training.
                           supported values include {"adam", "adagrad", "rmsprop"}
        lr: learning rate
        batch_size: batch size for training
        num_epochs: number of epochs for training
                              can be used when the threshold calculated is too low
        resume_train: parameter to decide if resume training is needed. Also,
                              based on this parameter the optimizer state dict
                              is stored in registry.

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
        resume_train: bool = False
    ):
        if not (model and seq_len):
            raise ValueError("No model and seq len provided!")
        if num_epochs < 1:
            raise ValueError("num_epochs must be a positive interger")

        self._model = model
        self.seq_len = seq_len
        self.criterion = self.init_criterion(loss_fn)
        self.optimizer = self.init_optimizer(optimizer, lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.resume_train = resume_train
        self._epochs_elapsed = 0

    @property
    def model_properties(self):
        model_properties_dict = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "epochs_elapsed": self._epochs_elapsed
        }
        if self.resume_train:
            model_properties_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        return model_properties_dict

    @property
    def model(self) -> AutoencoderModel:
        return self._model

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
        losses = []
        for epoch in range(1, self.num_epochs + 1):
            for x_batch in loader:
                self.optimizer.zero_grad()
                _, decoded = self._model(x_batch)
                loss = self.criterion(decoded, x_batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if epoch % log_freq == 0:
                _LOGGER.info(f"epoch : {epoch}, loss_mean : {np.mean(losses):.7f}")
            losses = []
            self._epochs_elapsed += 1
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

    def score(self, X: NDArray[float]) -> NDArray:
        r"""
        Returns the reconstruction error.

        Args:
            X: data

        Returns:
            numpy array with anomaly scores
        """
        x_recon = self.predict(X, seq_len=self.seq_len)
        recon_err = np.abs(X - x_recon)
        return recon_err

    def transform(self, X: NDArray[float]) -> NDArray:
        return self.score(X)

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
        if self.resume_train:
            self.optimizer.load_state_dict(metadata["optimizer_state_dict"])
            self._epochs_elapsed = metadata["epochs_elapsed"]
        self.num_epochs = metadata["num_epochs"]
        self.batch_size = metadata["batch_size"]

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
    r"""
    Class to simplify training, inference, loading and saving of Sparse Autoencoder.
    It inherits from AutoencoderPipeline class and serves as a wrapper around base network models.
    Sparse Autoencoder is a type of autoencoder that applies sparsity constraint.
    This helps in achieving information bottleneck even when the number of hidden units is huge.
    It penalizes the loss function such that only some neurons are activated at a time.
    This sparsity penalty helps in preventing overfitting.
    More details about Sparse Autoencoder can be found at
        <https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf>

    Note:
         This class only supports Pytorch models.
    Args:
        beta: regularization parameter (Defaults to 1e-3)
        rho: sparsity parameter value (Defaults to 0.05)
        method: regularization method
                        supported values include {"kl_div", "L1", "L2"}
                        (Defaults to "kl_div")
        model: model instance
        seq_len: sequence length
        loss_fn: loss function used for training
                        supported values include {"huber", "l1", "mse"}
        optimizer: optimizer to be used for training.
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
        resume_train: parameter to decide if resume training is needed. Also,
                              based on this parameter the optimizer state dict
                              is stored in registry.

    >>> # Example usage
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> from numalogic.models.autoencoder import SparseAEPipeline
    >>> x_train = np.random.randn(100, 3)
    >>> model = VanillaAE(signal_len=12, n_features=3)
    >>> sparse_ae_trainer = SparseAEPipeline(model=model, seq_len=36, num_epochs=30)
    >>> sparse_ae_trainer.fit(x_train)
    """

    def __init__(self, beta=1e-3, rho=0.05, method="kl_div", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.rho = rho
        self.reg_method = method

    def l1_loss(self) -> Tensor:
        r"""
        Loss function for computing sparse penalty based on L1 regularization.
        L1 regularization adds the absolute magnitude value of coefficient as the penalty term.

        Returns:
            Tensor
        """
        l1_lambda = self.beta
        l1_norm = sum(torch.linalg.norm(p, 1) for p in self._model.parameters())
        return torch.Tensor(l1_norm + l1_lambda)

    def l2_loss(self) -> Tensor:
        r"""
        Loss function for computing sparse penalty based on L2 regularization.
        L2 regularization adds the squared magnitude of coefficient as the penalty term.

        Returns:
            Tensor
        """
        l2_lambda = self.beta
        l2_norm = sum(torch.linalg.norm(p, 2) for p in self._model.parameters())
        return torch.Tensor(l2_norm + l2_lambda)

    def kl_divergence(self, activations: Tensor) -> Tensor:
        r"""
        Loss function for computing sparse penalty based on KL (Kullback-Leibler) Divergence.
        KL Divergence measures the difference between two probability distributions.

        Args:
            activations: encoded output from the model layer-wise

        Returns:
            Tensor
        """
        rho_hat = torch.mean(activations, dim=0)
        rho = torch.full(rho_hat.size(), self.rho)
        kl_loss = nn.KLDivLoss(reduction="sum")
        _dim = 0 if rho_hat.dim() == 1 else 1
        return kl_loss(torch.log_softmax(rho_hat, dim=_dim), torch.softmax(rho, dim=_dim))

    def calculate_regularized_loss(self, activation: Tensor) -> Tensor:
        r"""
        Loss Function to compute regularized loss penalty based on the chosen regularization method

        Args:
            activation: encoded output from the model layer-wise

        Raises:
            NotImplementedError: Unsupported regularization method value provided

        Returns:
            Tensor
        """
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
        r"""
        Fit function to train sparse autoencoder model

        Args:
            X: training dataset
            y: labels (Defaults to None)
            log_freq: frequency logging, i.e, number of epochs to be logged (Defaults to 5)
        """
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
