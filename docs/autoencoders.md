# Autoencoders

An Autoencoder is a type of Artificial Neural Network, used to learn efficient data representations (encoding) of unlabeled data.

It mainly consists of 2 components: an encoder and a decoder. The encoder compresses the input into a lower dimensional code, the decoder then reconstructs the input only using this code.

## Datamodules
Pytorch-lightning datamodules abstracts and separates the data functionality from the model and training itself.
Numalogic provides `TimeseriesDataModule` to help set up and load dataloaders.

```python
import numpy as np
from numalogic.tools.data import TimeseriesDataModule

train_data = np.random.randn(100, 3)
datamodule = TimeseriesDataModule(12, train_data, batch_size=128)
```

## Autoencoder Trainer

Numalogic provides a subclass of Pytorch-Lightning Trainer module specifically for Autoencoders.
This trainer provides a mechanism to train, validate and infer on data, with all the parameters supported by Lightning Trainer.

Here we are using `VanillaAE`, a Vanilla Autoencoder model.

```python
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.models.autoencoder import AutoencoderTrainer

model = VanillaAE(seq_len=12, n_features=3)
trainer = AutoencoderTrainer(max_epochs=50, enable_progress_bar=True)
trainer.fit(model, datamodule=datamodule)
```

## Autoencoder Variants

Numalogic supports 2 variants of Autoencoders currently.
More details can be found [here](https://www.deeplearningbook.org/contents/autoencoders.html).

### 1. Autoencoders

Basic autoencoders aim to find representations of the input data in a latent dimensional space.
Ideally, in order for the network to learn meaningful patterns, it is recommended that undercomplete
architectures are used, i.e. the latent space dimension being less than the input dimension.

Examples would be `VanillaAE`, `Conv1dAE`, `LSTMAE` and `TransformerAE`

### 2. Sparse autoencoders
A Sparse Autoencoder is a type of autoencoder that employs sparsity to achieve an information bottleneck.
Specifically the loss function is constructed so that activations are penalized within a layer.
So, by adding a sparsity regularization, we will be able to stop the neural network from copying the input and reduce overfitting.

Examples would be `SparseVanillaAE`, `SparseConv1dAE`, `SparseLSTMAE` and `SparseTransformerAE`

## Network architectures

Numalogic currently supports the following architectures.

#### Fully Connected

Vanilla Autoencoder model comprising only fully connected layers.

```python
from numalogic.models.autoencoder.variants import VanillaAE

model = VanillaAE(seq_len=12, n_features=2)
```

#### Convolutional

Conv1dAE is a 1D convolutional autoencoder.

The encoder network consists of convolutional layers and max pooling layers.
The decoder network tries to reconstruct the same input shape by corresponding transposed
convolutional and upsampling layers.

```python
from numalogic.models.autoencoder.variants import SparseConv1dAE

model = SparseConv1dAE(beta=1e-2, seq_len=12, in_channels=3, enc_channels=[8, 4])
```

#### LSTM

An LSTM (Long Short-Term Memory) Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture.

```python
from numalogic.models.autoencoder.variants import LSTMAE

model = LSTMAE(seq_len=12, no_features=2, embedding_dim=15)
```

#### Transformer

The transformer-based Autoencoder model was inspired from [Attention is all you need](https://arxiv.org/abs/1706.03762) paper.

It consists of an encoder and a decoder which are both stacks of residual attention blocks, i.e a stack of layers set in such a way that the output of a layer is taken and added to another layer deeper in the block.

These blocks can process an input sequence of variable length n without exhibiting a recurrent structure and allows transformer-based encoder-decoders to be highly parallelizable.

```python
from numalogic.models.autoencoder.variants import TransformerAE

model = TransformerAE(
    num_heads=8,
    seq_length=12,
    dim_feedforward=64,
    num_encoder_layers=3,
    num_decoder_layers=1,
)
```
