import os
import math
from time import time

import paddle as pd

import data
import config as cfg
from model import RNN


pd.device.set_device("gpu:0")
DATA = data.Poetries(cfg.batch)
LOSS = pd.nn.CrossEntropyLoss()
MODEL = RNN(
    cfg.num_layers, cfg.hidden_dim, cfg.embedding_dim, data.DICTIONARY_SIZE
)
MODEL_DIR = os.path.join("model", "parameters")
OPTIMIZER = pd.optimizer.Adam(cfg.lr, parameters=MODEL.parameters())


def log(head: str = "汉皇重色思倾国"):
    MODEL.eval()
    result = list(head)
    y, h = MODEL(pd.to_tensor([[data.encode[char] for char in head]]))
    while len(result) <= cfg.max_len:
        y = y[-1:, :]
        x = pd.argmax(y[0])
        result.append(data.decode[x.item()])
        if result[-1] == data.END:
            break
        y, h = MODEL(x.reshape((1, -1)), h)
    MODEL.train()
    with open("checkpoint.txt", "w") as output:
        output.write(str(epoch))
    with open("log.txt", "a", encoding="utf-8") as output:
        output.write(f"Epoch: {epoch + 1}\t")
        output.write("".join(result))


def progress():
    n = (epoch - EPOCH) * len(DATA) + min((step + 1) * cfg.batch, len(DATA))
    t = round(((cfg.epoch - EPOCH) * len(DATA) - n) / (n / (time() - TIME)))
    n = 100 * (n + EPOCH * len(DATA)) / (cfg.epoch * len(DATA))
    print("\rEpoch: {}/{} [{}>{}]{:.2f}% loss:{:.2f} eta {}:{:02}:{:02}".format(
        epoch + 1, cfg.epoch, "-" * (round(n) >> 1),
        "." * (50 - (round(n) >> 1)), n, loss.item(),
        t // 3600, (t % 3600) // 60, t % 60
    ), end=" " * 4)


if "last.pd""params" in os.listdir(MODEL_DIR):
    EPOCH = int(open(os.path.join(MODEL_DIR, "checkpoint.txt")).read())
    MODEL.load_dict(pd.load(os.path.join(MODEL_DIR, "last.pd""params")))
else:
    EPOCH = 0
TIME = time()

for epoch in range(EPOCH + 1, cfg.epoch):
    OPTIMIZER.set_lr(cfg.lr * math.cos(0.5 * epoch * math.pi / cfg.epoch))
    for step, poetries in enumerate(DATA):
        poetries = pd.to_tensor(poetries)
        loss = LOSS(
            MODEL(poetries[:, :-1])[0],
            poetries[:, 1:].flatten()
        )
        progress()
        loss.backward()
        OPTIMIZER.step()
        OPTIMIZER.clear_grad()
    pd.save(
        MODEL.state_dict(),
        os.path.join(MODEL_DIR, "last.pd""params")
    )
    log()
pd.save(MODEL.state_dict(), os.path.join(MODEL_DIR, "final.pd""params"))
