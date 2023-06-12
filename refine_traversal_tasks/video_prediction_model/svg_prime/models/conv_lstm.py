import torch
import torch.nn as nn
from .convlstm_cell import ConvLSTMCell
from torch.autograd import Variable

class ConvLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, image_size, n_layers, batch_size, expand=1):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.image_size = image_size
        self.embed = nn.Conv2d(input_size, int(hidden_size*expand), 3, 1, 1) # Initial embedding conv layer
        self.lstm = nn.ModuleList([ConvLSTMCell(int(hidden_size*expand), int(hidden_size*expand), (3, 3), True) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Conv2d(int(hidden_size*expand), output_size, 3, 1, 1),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for layer in self.lstm:
            hidden.append(layer.init_hidden(self.batch_size, self.image_size))
        return hidden

    def forward(self, input):
        embedded = self.embed(input)
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        return self.output(h_in)


class ConvGaussianLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, image_size, n_layers, batch_size, expand=1):
        super(ConvGaussianLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Conv2d(input_size, int(hidden_size*expand), 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(int(hidden_size*expand), int(hidden_size*expand), (3, 3), True) for _ in range(self.n_layers)])
        self.mu_net = nn.Conv2d(int(hidden_size*expand), output_size, 3, 1, 1)
        self.logvar_net = nn.Conv2d(int(hidden_size*expand), output_size, 3, 1, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for layer in self.lstm:
            hidden.append(layer.init_hidden(self.batch_size, self.image_size))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input)
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar