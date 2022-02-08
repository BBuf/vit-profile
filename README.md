# vit-bench


## Install

### pytorch

```shell
CUDA_VERSION='cu112'
pip install torch==1.10.1+${CUDA_VERSION}
```

### oneflow

```shell
CUDA_VERSION='cu114'
pip install oneflow==0.6.0+${CUDA_VERSION} --find-links https://release.oneflow.info
```


### 3090Ti 单卡速度测试

- BatchSize = 64

| 框架模式      | Iter数 | 训练速度 |
| ------------- | ------ | -------- |
| PyTorch       | 200    | 63s      |
| OneFlow Eager | 200    | 61s      |
| OneFlow Graph | 200    | 60s      |
