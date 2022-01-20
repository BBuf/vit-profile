from typing import Callable

import numpy as np
from lib.vit import ViT_B_16_224
import oneflow as flow
from oneflow import nn
from tqdm import tqdm, trange


def bench(forward_and_backward: Callable, x, y, n=1000):
    batch_size = x.shape[0]
    device = flow.device('cuda')
    x_of = flow.tensor(x)
    y_of = flow.tensor(y)
    x_of = x_of.to(device)
    y_of = y_of.to(device)

    with tqdm(total=n * batch_size) as pbar:
        for _ in range(n):

            loss, output = forward_and_backward(x_of, y_of)

            loss.item()

            pbar.update(batch_size)


class VitTrainGraph:

    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def __call__(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, y_pred


def main():
    batch_size = 64

    np.random.seed(42)

    device = flow.device('cuda')
    vit = ViT_B_16_224()

    # from lib.timm_vit import vit_base_patch16_224_bench
    # vit = vit_base_patch16_224_bench()

    # vit = flow.jit.script(vit)

    vit.to(device)

    optimizer = flow.optim.SGD(vit.parameters(), lr=0.001)

    x = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    y = np.random.randint(0, 1000, (batch_size,))

    model_graph = VitTrainGraph(vit, optimizer)

    # bench(model_graph, x, y, n=10)

    bench(model_graph, x, y, n=100)


if __name__ == '__main__':
    main()
