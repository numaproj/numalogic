import random


class SparsityGenerator:
    """
    Introduces sparsity to data by reassigning certain rows and columns
    in the dataframe to value of 0 (based on sparsity ratio).
    """

    def __init__(self, data, sparse_ratio=0.2):
        """
        @param data: Reference Multivariate time series DataFrame
        @param sparse_ratio: Ratio of sparsity to introduce wrt
            to number of samples
        """
        self.sparse_ratio = sparse_ratio
        self._data = data

    def generate_sparsity(self):
        shape = self._data.shape

        # based on sparsity ratio generating the rows
        # to which the data is going to be imputed with 0
        rows = random.sample(range(0, shape[0]), int(shape[0] * self.sparse_ratio))

        for row in rows:

            # identifying the columns to which the data is going to be imputed with 0.
            columns = random.sample(range(0, shape[1]), int(shape[1] * self.sparse_ratio))
            self._data.iloc[row, columns] = 0

    @property
    def data(self):
        return self._data
