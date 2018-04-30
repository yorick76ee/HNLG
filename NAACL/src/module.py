import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(
        self,
        en_vocab_size,
        hidden_size,
        n_layers,
        bidirectional
    ):
        super(EncoderRNN, self).__init__()
        self.vocab_size = en_vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
                self.vocab_size, hidden_size,
                n_layers, batch_first=True,
                bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        size = inputs.size()
        embedded = torch.Tensor(size[0], size[1], self.vocab_size).zero_()
        inputs = inputs.data.unsqueeze(2)
        embedded.scatter_(dim=2, index=inputs.cpu(), value=1.)
        embedded = Variable(embedded.cuda())
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(
                torch.zeros(
                    self.n_layers * (self.bidirectional + 1),
                    batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def initAttrHidden(self, inputs, batch_size):
        # use attributes (encoder input) as RNN initial state
        size = inputs.size()
        attrs = torch.Tensor(size[0], size[1], self.vocab_size).zero_()
        inputs = inputs.data.unsqueeze(2)
        attrs.scatter_(dim=2, index=inputs.cpu(), value=1.)

        # compress attrs into a initial state vector:
        # (batch_size, seq_length, num_attr) -> (batch_size, num_attr)
        attrs = attrs.sum(dim=1)
        # trim _UNK and _PAD
        attrs[:, 0:2] = 0
        # pad zeros
        attrs = torch.cat(
                [
                    attrs,
                    torch.zeros(batch_size, self.hidden_size - self.vocab_size)
                ], 1)
        attrs = attrs.repeat(self.n_layers * (self.bidirectional + 1), 1, 1)
        return Variable(attrs.cuda()) if use_cuda else Variable(attrs)


class Attn(nn.Module):
    def __init__(
            self, method, en_hidden_size, de_hidden_size,
            n_en_layers, n_de_layers, bidirectional):
        super(Attn, self).__init__()

        self.method = method
        self.n_en_layers = n_en_layers
        self.n_de_layers = n_de_layers
        en_hidden_size = en_hidden_size * n_en_layers * (bidirectional + 1)
        de_hidden_size = de_hidden_size * n_de_layers * (bidirectional + 1)

        # If en_layers != de_layers,
        # then the dot attention is same as general attention
        if self.method == "dot" and self.n_en_layers != self.n_de_layers:
            self.method = 'general'

        if self.method == 'general':
            self.attn = nn.Linear(en_hidden_size, de_hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(en_hidden_size + de_hidden_size, 1)

    def forward(self, hidden, encoder_hiddens):
        attn_energies = self.score(hidden, encoder_hiddens)
        return torch.nn.Softmax(dim=1)(attn_energies)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            encoder_output = encoder_output.permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * encoder_output).sum(1)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output).permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * energy).sum(1)
            return energy

        elif self.method == 'concat':
            seq_length = encoder_output.size(1)
            hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
            energy = self.attn(
                    torch.cat((hidden, encoder_output), 2)).squeeze(2)
            return energy


class DecoderRNN(nn.Module):
    def __init__(
            self,
            embedding,
            de_vocab_size,
            de_embedding_dim,
            en_hidden_size,
            de_hidden_size,
            n_en_layers=1,
            n_de_layers=1,
            attn_method='concat',
            bidirectional=False,
            feed_last=False,
    ):
        super(DecoderRNN, self).__init__()
        self.vocab_size = de_vocab_size
        self.embedding_dim = \
            de_embedding_dim * 2 if feed_last else de_embedding_dim
        self.hidden_size = de_hidden_size
        self.n_layers = n_de_layers
        self.bidirectional = bidirectional

        if attn_method != 'none':
            rnn_dim = self.embedding_dim + en_hidden_size * (bidirectional + 1)
            self.attn = Attn(
                    attn_method, en_hidden_size, de_hidden_size,
                    n_en_layers, n_de_layers, bidirectional)
        else:
            rnn_dim = self.embedding_dim
            self.attn = None

        self.embedding = embedding
        self.rnn = nn.GRU(
                rnn_dim, de_hidden_size, n_de_layers,
                batch_first=True, bidirectional=bidirectional)

        # to handle encoder decoder hidden_size mismatch
        self.transform_layer = nn.Linear(en_hidden_size, de_hidden_size)

        if bidirectional:
            self.out = nn.Linear(de_hidden_size * 2, self.vocab_size)
        else:
            self.out = nn.Linear(de_hidden_size, self.vocab_size)

        self.feed_last = feed_last

    def forward(
            self, input, last_hidden, encoder_hiddens, last_output=None):

        embedded = self.embedding(input)
        if self.feed_last and last_output is not None:
            embedded = torch.cat((embedded, self.embedding(last_output)), 2)
        if self.attn:
            batch_size = last_hidden.size(1)
            attn_weights = self.attn(
                    last_hidden.permute(1, 0, 2).contiguous().view(
                        batch_size, -1),
                    encoder_hiddens).unsqueeze(2)
            attn = (attn_weights * encoder_hiddens).sum(1).unsqueeze(1)
            rnn_input = torch.cat((embedded, attn), 2)
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, last_hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(
                torch.zeros(
                    self.n_layers * (self.bidirectional + 1),
                    batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
