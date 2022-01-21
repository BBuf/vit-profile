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


### 3090Ti 单卡速度测试

| 框架模式 | Iter数 |速度 | 预热轮次|
| -- | -- | -- | -- |
| PyTorch | 200 | 63s | 0 |
| OneFlow Eager | 200 | 68s | 0|
| OneFlow Graph | 200 | 70s | 0 |
| PyTorch | 200 | 63s | 5 |
| OneFlow Eager | 200 | 63s | 5|
| OneFlow Graph | 200 | 63s | 5 |

