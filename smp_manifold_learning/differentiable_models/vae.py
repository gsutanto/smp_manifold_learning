# File: vae.py
# An implementation of Variational Auto-Encoder
# Modified from https://github.com/pytorch/examples/blob/master/vae/main.py
# References:
# [1] Kingma and Welling. Auto-Encoding Variational Bayes. ICLR 2014.
#     https://arxiv.org/abs/1312.6114
# [2] Doersch. Tutorial on Variational Autoencoders. 2016.
#     https://arxiv.org/pdf/1606.05908.pdf
#
import copy
import torch

import smp_manifold_learning.differentiable_models.nn as nn
import smp_manifold_learning.differentiable_models.utils as utils


class VAE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_hidden_sizes,
                 latent_dim,
                 activation='tanh',
                 use_batch_norm=True,
                 drop_p=0.0,
                 name=''):
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

        self.encoder_function = nn.LNMLP(
            self._input_dim,
            self.encoder_hidden_sizes,
            2 * self._latent_dim,  # mu and logvar
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
        z_mu_and_logvar = self.encoder_function(input_tensor)
        z_mu = z_mu_and_logvar[:, :self._latent_dim]
        z_logvar = z_mu_and_logvar[:, self._latent_dim:]
        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + (eps * z_std)

    def decode(self, latent_tensor):
        return self.decoder_function(latent_tensor)

    def forward_full(self, input_tensor):
        z_mu, z_logvar = self.encode(input_tensor)
        z = self.reparameterize(z_mu, z_logvar)
        return self.decode(z), z_mu, z_logvar

    def forward(self, input_tensor):
        [recon_tensor, _, _] = self.forward_full(input_tensor)
        return recon_tensor

    def get_loss_components(self, recon_x, x, z_mu, z_logvar):
        # Reconstruction Error:
        #RE = utils.compute_mse(recon_x, x)  # averaged over batch
        RE = utils.compute_nmse_loss(recon_x, x)  # averaged over batch

        # KL Divergence:
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)  # averaged over batch
        KLD = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        return RE, KLD

    def sample(self):
        z = torch.randn((1, self._latent_dim))
        return self.decode(z).detach().numpy()[0]
