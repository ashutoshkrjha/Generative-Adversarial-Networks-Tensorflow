# Generative-Adversarial-Networks-Tensorflow
A Tensorflow Implementation of Generative Adversarial Networks as presented in the original paper by Goodfellow et. al. (https://arxiv.org/abs/1406.2661)

## Introduction:
The code runs a toy GAN framework to try to generate samples from a known 1D gaussian distribution using uniform random noise. The algorithm used is the same as Algorithm 1 in the paper with the loss function of the generator changed to maximize log(D(G(z))) as suggested in the text of the paper. 

## Implementation Details:
Both the generator and the discriminator networks are 3 layer MLP's (Multi-layer perceptrons) with one input, one output and one hidden layer.
The optimizer used to train the networks is Adam.

## Usage:
To use, run:
```
$ python gan.py [parameters]
```

#### Parameters:
'-mean'     : Mean of target gaussian (float)
'-std'      : Standard Deviation of Target Gaussian (float)
'-hneurons' : Number of hidden layer neurons to use (int)
'-epoch'    : Number of epochs to train (int)
'-minbatch' : Size of batch to train (int)
'-sample'   : Size of points to sample from true distribution (int)
