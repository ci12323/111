from random import random

import paddle as pd

import data
import config as cfg
from work.model import RNN

__all__ = ["Poet"]


class Poet:
    def __init__(self):
        self.rnn = RNN(
            cfg.num_layers,
            cfg.hidden_dim,
            cfg.embedding_dim,
            data.DICTIONARY_SIZE
        )
        self.rnn.eval()
        self.rnn.load_dict(
            pd.load("C:/Users/M/1111/work/model/parameters/final.pdparams")
        )

    def renewal(self, head: str) -> str:
        poetry = list(head)
        y, h = self.rnn(pd.to_tensor([[data.encode[char] for char in head]]))
        while len(poetry) <= cfg.max_len:
            y = y[-1:, :]
            x = pd.argmax(y[0])
            char = data.decode[x.item()]
            if char == data.END:
                break
            poetry.append(char)
            y, h = self.rnn(x.reshape((1, -1)), h)
        return "".join(poetry)

    def acrostic(self, head: str) -> str:
        poetry, h = [], None
        punctuations = "，；。！？"
        if len(head) % 2 or random() < 0.5:
            punctuations = punctuations[2:]
        for char in head:
            poetry.append(char)
            y, h = self.rnn(pd.to_tensor([[data.encode[char]]]), h)
            while poetry[-1] not in punctuations:
                x = pd.argmax(y[0])
                poetry.append(data.decode[x.item()])
                y, h = self.rnn(x.reshape((1, -1)), h)
            poetry.append("\n")
        return "".join(poetry)
