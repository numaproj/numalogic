# Autoencoders

An Autoencoder is a type of Artificial Neural Network, used to learn efficient data representations (encoding) of unlabeled data. 

It mainly consist of 2 components: an encoder and a decoder. The encoder compresses the input into a lower dimensional code, the decoder then reconstructs the input only using this code.

### Autoencoder Pipelines

Numalogic provides two types of pipelines for Autoencoders. These pipelines serve as a wrapper around the base network models, making it easier to train, predict and generate scores. Also, this module follows the sklearn API.

#### AutoencoderPipeline

Here we are using `VanillAE`, a Vanilla Autoencoder model.

```python 
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.autoencoder import SparseAEPipeline

model = AutoencoderPipeline(
    model=VanillaAE(signal_len=12, n_features=3), seq_len=seq_len
)
model.fit(X_train)
```

#### SparseAEPipeline

A Sparse Autoencoder is a type of autoencoder that employs sparsity to achieve an information bottleneck. Specifically the loss function is constructed so that activations are penalized within a layer.

So, by adding a sparsity regularization, we will be able to stop the neural network from copying the input and reduce overfitting.

```python 
from numalogic.models.autoencoder.variants import Conv1dAE
from numalogic.models.autoencoder import SparseAEPipeline

model = SparseAEPipeline(
    model=VanillaAE(signal_len=12, n_features=3), seq_len=36, num_epochs=30
)
model.fit(X_train)
```

### Autoencoder Variants

Numalogic supports the following variants of Autoencoders

#### VanillaAE

Vanilla Autoencoder model comprising only fully connected layers.

```python
from numalogic.models.autoencoder.variants import VanillaAE

model = VanillaAE(seq_len=12, n_features=2)
```   

#### Conv1dAE

Conv1dAE is a one dimensional Convolutional Autoencoder with multichannel support.
   
```python
from numalogic.models.autoencoder.variants import Conv1dAE

model=Conv1dAE(in_channels=3, enc_channels=8)
```

#### LSTMAE

An LSTM (Long Short-Term Memory) Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture.

```python
from numalogic.models.autoencoder.variants import LSTMAE

model = LSTMAE(seq_len=12, no_features=2, embedding_dim=15)

```

#### TransformerAE

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