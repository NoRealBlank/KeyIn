import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden
    
    def cond_hidden(self, cond):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((cond, cond))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded

        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class KeyValue_lstm(nn.Module):
    def __init__(self, input_size, output_key_size, output_value_size, hidden_size, n_layers, batch_size):
        super(KeyValue_lstm, self).__init__()
        self.input_size = input_size
        self.output_key_size = output_key_size
        self.output_value_size = output_value_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.key_net = nn.Linear(hidden_size, output_key_size)
        self.value_mu_net = nn.Linear(hidden_size, output_value_size)
        self.value_logvar_net = nn.Linear(hidden_size, output_value_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        key = self.key_net(h_in)
        value_mu = self.value_mu_net(h_in)
        value_logvar = self.value_logvar_net(h_in)
        return key, value_mu, value_logvar

class keyframe_lstm(nn.Module):
    def __init__(self, input_size, output_size, delta_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.delta_net = nn.Sequential(
                nn.Linear(hidden_size, delta_size),
                nn.Softmax())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden
    
    def cond_hidden(self, cond):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((cond, cond))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded

        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in), self.delta_net(h_in)

import torch.nn.functional as F

class EmbeddingMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(EmbeddingMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

class KeyValueAttention(torch.nn.Module):
    def __init__(self, key_dim, value_dim):
        super(KeyValueAttention, self).__init__()
        # Assume query_dim is the same as key_dim
        self.scale = 1.0 / (key_dim ** 0.5)

    def forward(self, query, keys, values):
        """
        query: Tensor of shape [batch_size, query_dim]
        (embedding of the last keyframe)
        
        keys: Tensor of shape [batch_size, num_keys, key_dim]
        (output embedding of inference network for all frames)
        
        values: Tensor of shape [batch_size, num_keys, value_dim]
        (output value of the inference network for all frames)
        """
        # Compute the dot products
        # query shape after unsqueeze: [batch_size, 1, query_dim]
        # keys shape: [batch_size, num_keys, key_dim]
        # scores shape: [batch_size, 1, num_keys]
        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)) * self.scale

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Multiply the weights by the values
        # attn_weights shape: [batch_size, 1, num_keys]
        # values shape: [batch_size, num_keys, value_dim]
        # output shape: [batch_size, 1, value_dim]
        output = torch.matmul(attn_weights, values)

        # Squeeze to remove the middle dimension
        return output.squeeze(1)
