from paddle import nn


__all__ = ["RNN"]


class RNN(nn.Layer):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 dictionary_size: int):
        super(RNN, self).__init__()
        self.out = nn.Linear(hidden_dim, dictionary_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.embedding = nn.Embedding(dictionary_size, embedding_dim)

    def forward(self, sequence, hidden=None):
        batch, length = sequence.shape
        output, hidden = self.lstm(self.embedding(sequence), hidden)
        return self.out(output.reshape((batch * length, -1))), hidden
