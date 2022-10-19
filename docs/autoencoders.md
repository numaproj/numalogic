# Autoencoders:

An Autoencoder is a type of Artificial Neural Network, used to learn efficient data representations (encoding) of unlabeled data. 

It mainly consist of 2 components: an encoder and a decoder. The encoder compresses the input into a lower dimensional code, the decoder then reconstructs the input only using this code.

### Autoencoder Pipelines:

Numalogic provides two types of pipelines for Autoencoders.

#### AutoencoderPipeline:

Here we are using `VanillAE`, a Vanilla Autoencoder model comprising only fully connected layers.

```python 
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.autoencoder.pipeline import SparseAEPipeline

model = AutoencoderPipeline(
    model=VanillaAE(signal_len=12, n_features=3), seq_len=seq_len
)
model.fit(X_train)
```

#### SparseAEPipeline:

`SparseAEPipeline` can be used when the training data is sparse. 

Here we are using `Conv1dAE`, a one dimensional Convolutional Autoencoder with multichannel support.

```python 
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.autoencoder.pipeline import SparseAEPipeline

model = SparseAEPipeline(
    model=Conv1dAE(in_channels=3, enc_channels=8), seq_len=36, num_epochs=30
)
model.fit(X_train)
```

### Autoencoder Variants:

Numalogic supports the following variants of Autoencoders

#### VanillaAE

```python
from numalogic.models.autoencoder.variants import VanillaAE

model = VanillaAE(seq_len=12, n_features=2)
```   

#### Conv1dAE
   
```python
from numalogic.models.autoencoder.variants import Conv1dAE

model=Conv1dAE(in_channels=3, enc_channels=8)
```

#### LSTMAE

```python
from numalogic.models.autoencoder.variants import LSTMAE

model = LSTMAE(seq_len=12, no_features=2, embedding_dim=15)

```
