from typing import Callable

import time
import datetime

import numpy as np
from lib.torch_vit import vit_base_patch16_224
import torch
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm, trange


def bench(forward: Callable, x, y, n=1000):
    batch_size = x.shape[0]
    device = torch.device('cuda')
    x_of = torch.tensor(x)
    y_of = torch.tensor(y)
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
        torch.cuda.nvtx.range_push('forward')
        y_pred = self.model(x)
        torch.cuda.nvtx.range_pop()
        return y_pred


def main():
    batch_size = 64

    np.random.seed(42)

    cudnn.benchmark = True
    device = torch.device('cuda')
    vit = vit_base_patch16_224(num_classes=1000)

    # from lib.timm_vit import vit_base_patch16_224_bench
    # vit = vit_base_patch16_224_bench()

    # vit = torch.jit.script(vit)

    vit.to(device)


    x = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    y = np.random.randint(0, 1000, (batch_size,))

    model_graph = VitEvalGraph(vit)

    # bench(model_graph, x, y, n=10)

    bench(model_graph, x, y, n=200)


if __name__ == '__main__':
    main()
