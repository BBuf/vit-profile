# vit-bench


## Install

### pytorch

```shell
CUDA_VERSION='cu112'
pip install torch==1.10.1+${CUDA_VERSION}
```

### oneflow

```shell
CUDA_VERSION='cu112'
pip install oneflow==0.6.0+${CUDA_VERSION} --find-links https://release.oneflow.info
```


### 3090Ti 单卡速度测试 （实验1）

- BatchSize = 64

| 框架模式 | Iter数 |训练速度 | 推理速度 |
| -- | -- | -- | -- |
| PyTorch | 200 | 63s | 22s |
| OneFlow Eager | 200 | 77s | 20s |


### 3090Ti 单卡速度测试（实验2）

- 去掉预热和训练循环的loss.item同步
- BatchSize = 64

| 框架模式 | Iter数 |训练速度 | 推理速度 |
| -- | -- | -- | -- |
| PyTorch | 200 | 63s | 22s |
| OneFlow Eager | 200 | 74s | 20s |

### 3090Ti 单卡速度测试（实验3）

- 去掉训练循环的loss.item同步（预热阶段的loss.item保留）
- BatchSize = 64

| 框架模式 | Iter数 |训练速度 | 推理速度 |
| -- | -- | -- | -- |
| PyTorch | 200 | 63s | 22s |
| OneFlow Eager | 200 | 63s | 20s |


### 3090Ti 单卡速度测试 （实验4）

- BatchSize = 32

| 框架模式 | Iter数 |训练速度 | 推理速度 |
| -- | -- | -- | -- |
| PyTorch | 200 | 32s | 11s |
| OneFlow Eager | 200 | 33s | 11s |
