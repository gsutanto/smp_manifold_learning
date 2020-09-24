# File: autoencoder.py
#
import copy
import torch

import smp_manifold_learning.differentiable_models.nn as nn


class AutoEncoder(torch.nn.Module):

    def __init__(self, input_dim, encoder_hidden_sizes, latent_dim, activation='tanh',
                 use_batch_norm=True, drop_p=0.0, name=''):
        """
        Arguments:
        - `input_dim`: int
        - `encoder_hidden_sizes`: [list]
        - `latent_dim`: int
        """
        self._input_dim = input_dim
        self._latent_dim = latent_dim

        super().__init__()

        self.name = name

        self.encoder_hidden_sizes = encoder_hidden_sizes

        # decoder's hidden layer topology is a mirror of encoder's hidden layer topology
        self.decoder_hidden_sizes = copy.deepcopy(encoder_hidden_sizes)
        self.decoder_hidden_sizes.reverse()

        self.encoder_function = nn.LNMLP(self._input_dim, 
                                         self.encoder_hidden_sizes, 
                                         self._latent_dim,
                                         activation=activation, 
                                         use_batch_norm=use_batch_norm, 
                                         drop_p=drop_p)
        self.decoder_function = nn.LNMLP(self._latent_dim, 
                                         self.decoder_hidden_sizes, 
                                         self._input_dim,
                                         activation=activation, 
                                         use_batch_norm=use_batch_norm, 
                                         drop_p=drop_p)

    def encode(self, input_tensor):
        return self.encoder_function(input_tensor)

    def decode(self, latent_tensor):
        return self.decoder_function(latent_tensor)

    def forward(self, input_tensor):
        return self.decode(self.encode(input_tensor))
