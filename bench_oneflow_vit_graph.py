from typing import Callable

import numpy as np
import oneflow
from oneflow import nn
from tqdm import tqdm, trange

from lib.vit import ViT_B_16_224


def bench(forward_and_backward: Callable, x, y, n=1000):
    batch_size = x.shape[0]
    device = oneflow.device('cuda')

    x_of = oneflow.tensor(x)
    y_of = oneflow.tensor(y)
    x_of = x_of.to(device)
    y_of = y_of.to(device)

    with tqdm(total=n * batch_size) as pbar:
        for _ in range(n):

            loss, output = forward_and_backward(x_of, y_of)

            loss.item()

            pbar.update(batch_size)


class VitTrainGraph(nn.Graph):

    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.config.allow_fuse_add_to_output()
        self.config.allow_fuse_model_update_ops()

        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        return loss, y_pred


def main():
    batch_size = 64

    np.random.seed(42)

    device = oneflow.device('cuda')
    vit = ViT_B_16_224()
    # from flowvision.models.vit import vit_b_16_224
    # vit = vit_b_16_224()
    vit.to(device)

    optimizer = oneflow.optim.SGD(vit.parameters())

    x = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    y = np.random.randint(0, 1000, (batch_size,))

    model_graph = VitTrainGraph(vit, optimizer)

    # model_graph.debug()

    bench(model_graph, x, y, n=10)

    bench(model_graph, x, y, n=100)


if __name__ == '__main__':
    main()
