from typing import Callable

import time
import datetime

import numpy as np
from lib.vit import vit_base_patch16_224
import oneflow as flow
from oneflow import nn
from tqdm import tqdm, trange


def bench(forward: Callable, x, y, n=1000):
    batch_size = x.shape[0]
    device = flow.device('cuda')
    x_of = flow.tensor(x)
    y_of = flow.tensor(y)
    x_of = x_of.to(device)
    y_of = y_of.to(device)

    #warm up
    for _ in range(5):
        output = forward(x_of, y_of)

    start_time = time.time()
    
    for _ in range(n):
        output = forward(x_of, y_of)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(total_time_str)

class VitEvalGraph:

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x, y):
        flow._oneflow_internal.profiler.RangePush('forward')
        y_pred = self.model(x)
        flow._oneflow_internal.profiler.RangePop()
        return y_pred


def main():
    batch_size = 32

    np.random.seed(42)

    device = flow.device('cuda')
    vit = vit_base_patch16_224()

    # from lib.timm_vit import vit_base_patch16_224_bench
    # vit = vit_base_patch16_224_bench()

    # vit = flow.jit.script(vit)

    vit.to(device)


    x = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    y = np.random.randint(0, 1000, (batch_size,))

    model_graph = VitEvalGraph(vit)

    # bench(model_graph, x, y, n=10)

    bench(model_graph, x, y, n=200)


if __name__ == '__main__':
    main()
