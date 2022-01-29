from typing import Callable

import time
import datetime

import numpy as np
import oneflow
from oneflow import nn
from tqdm import tqdm, trange

from lib.vit import vit_base_patch16_224


def bench(forward: Callable,  n=1000):
    #warm up
    for _ in range(5):
        output = forward()

    start_time = time.time()
    for _ in range(n):
        output = forward()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(total_time_str)

class VitEvalGraph(nn.Graph):

    def __init__(self, model):
        super().__init__()
        self.model = model


    def build(self):
        batch_size = 64
        x = oneflow.rand(batch_size, 3, 224, 224).to(oneflow.device('cuda'))
        y = oneflow.randint(0, 1000, (batch_size,)).to(oneflow.device('cuda'))

        y_pred = self.model(x)
        return y_pred


def main():
    batch_size = 64

    np.random.seed(42)

    device = oneflow.device('cuda')
    vit = vit_base_patch16_224()
    # from flowvision.models.vit import vit_b_16_224
    # vit = vit_b_16_224()
    vit.to(device)

    model_graph = VitEvalGraph(vit)

    # model_graph.debug()

    # bench(model_graph, n=10)

    bench(model_graph, n=200)


if __name__ == '__main__':
    main()
