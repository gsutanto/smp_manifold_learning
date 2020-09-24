# File: nn.py
# Modified from https://github.com/sisl/mechamodlearn/blob/master/mechamodlearn/nn.py
#
import torch


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


ACTIVATIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'elu': torch.nn.ELU,
    'identity': Identity
}


class LNMLP(torch.nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh', gain=1.0,
                 use_layer_norm=False, use_batch_norm=False, drop_p=0.0):
        self._hidden_sizes = hidden_sizes
        self._gain = gain
        self._use_layer_norm = use_layer_norm

        super().__init__()

        if isinstance(activation, str):
            activation = [activation] * len(hidden_sizes)
        else:  # if isinstance(activation, list):
            activation = list(activation)
            assert (len(activation) == len(hidden_sizes))

        self.topology = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(self.topology) - 2):
            # linear layer: output = (weights * input) + bias
            layers.append(torch.nn.Linear(self.topology[i], self.topology[i + 1]))

            if use_batch_norm:
                # batch-normalization layer
                layers.append(torch.nn.BatchNorm1d(self.topology[i + 1]))

            # activation function (tanh/ReLU/etc.) layer
            layers.append(ACTIVATIONS[activation[i]]())

            # layer-normalization layer
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(self.topology[i + 1]))

            # dropout layer
            layers.append(torch.nn.Dropout(p=drop_p))

        # the output layer is only linear layer (this can be modified of course)
        layers.append(torch.nn.Linear(self.topology[-2], self.topology[-1]))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)
        self.reset_params(gain=gain)

    def forward(self, inp):
        return self.mlp(inp)

    def reset_params(self, gain=1.0):
        self.apply(lambda x: weights_init_mlp(x, gain=gain))


def weights_init_mlp(m, gain=1.0):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_normc_(weight, gain=1.0):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
