from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin, BaseEstimator


class DataIndependentTransformers(TransformerMixin, BaseEstimator):
    """Base class for stateless transforms."""

    def fit(self, _: ArrayLike):
        return self
