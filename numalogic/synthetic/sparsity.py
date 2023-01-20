# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
