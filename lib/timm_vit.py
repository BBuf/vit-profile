from timm.models.vision_transformer import Attention, vit_base_patch16_224
from torch.nn import SyncBatchNorm
from torch import nn, Tensor
import torch



def vit_base_patch16_224_bench(num_classes: int = 1000):
    vit = vit_base_patch16_224(num_classes=num_classes)
    vit = convert_multihead_attention(vit)
    return vit


class MultiheadAttention(nn.MultiheadAttention):

    def forward(self, x: Tensor):
        pass


def convert_multihead_attention(module: nn.Module):
    module_output = module
    if isinstance(module, Attention):
        print(f'replace {module.__class__}')
        module_output = torch.jit.script(module)
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
        del module
        return module_output
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_multihead_attention(child)
        )
    del module
    return module_output
