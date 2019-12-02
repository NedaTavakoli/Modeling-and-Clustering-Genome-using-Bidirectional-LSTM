import torch
import torch.nn as nn
from torch.autograd import Variable  # Variable has been depricated.  Now it just creates a tensor
# https://pytorch.org/docs/stable/autograd.html#variable-deprecated


class LM_LSTM(nn.Module):
    """Simple LSMT-based language model"""
    def __init__(self, hidden_dim, embedding_dim, num_steps, batch_size, vocab_size, num_layers, dp_keep_prob,
                 bidirectional):
        super(LM_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dp_keep_prob = dp_keep_prob
        self.num_layers = num_layers
        self.dropout = nn.Dropout(1 - dp_keep_prob)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=1 - dp_keep_prob,
                            bidirectional=bidirectional)
        if self.bidirectional:
            self.num_directions = 2  # The number of directions of this LSTM (Bidirections is 2)
        else:
            self.num_directions = 1  # The number of directions of this LSTM (One direction is 1)
        self.linear_input = self.num_directions * self.hidden_dim
        self.sm_fc = nn.Linear(in_features=self.linear_input,
                               out_features=vocab_size)
        self.init_weights()
        self.direction = None

    def init_weights(self):
        init_range = 0.1
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.sm_fc.bias.data.fill_(0.0)
        self.sm_fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_dim).zero_()))

    def forward(self, inputs, hidden, num_steps=None, batch_size=None):
        if num_steps is None:
            num_steps = self.num_steps
        if batch_size is None:
            batch_size = self.batch_size
        embeds = self.dropout(self.word_embeddings(inputs))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.sm_fc(lstm_out.view(-1, self.linear_input))
        output = logits.view(num_steps, batch_size, self.vocab_size)
        return output, hidden


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor: #Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
