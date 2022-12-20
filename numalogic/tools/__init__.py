from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin, BaseEstimator


class DataIndependentTransformers(TransformerMixin, BaseEstimator):
    def fit(self, _: ArrayLike):
        return self
