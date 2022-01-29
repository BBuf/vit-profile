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

| 框架模式 | Iter数 |训练速度 | 推理速度 |
| -- | -- | -- | -- |
| PyTorch | 200 | 63s | 22s |
| OneFlow Eager | 200 | 77s | 20s |



