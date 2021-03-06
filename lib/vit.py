"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from flowvision.layers.blocks import PatchEmbed
from flowvision.layers.regularization import DropPath
# from flowvision.layers.weight_init import trunc_normal_, lecun_normal_
from flowvision.models.utils import load_state_dict_from_url
from flowvision.models.registry import ModelCreator

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

model_urls = {
    "vit_tiny_patch16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_tiny_patch16_224.zip",
    "vit_tiny_patch16_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_tiny_patch16_384.zip",
    "vit_small_patch32_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_small_patch32_224.zip",
    "vit_small_patch32_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_small_patch32_384.zip",
    "vit_small_patch16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_small_patch16_224.zip",
    "vit_small_patch16_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_small_patch16_384.zip",
    "vit_base_patch32_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_base_patch32_224.zip",
    "vit_base_patch32_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_base_patch32_384.zip",
    "vit_base_patch16_224": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_base_patch16_224.zip",
    "vit_base_patch16_384": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/VisionTransformer/vit_base_patch16_384.zip",
}

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query = nn.Linear(dim, self.num_heads * self.head_dim)
        self.key = nn.Linear(dim, self.num_heads * self.head_dim)
        self.value = nn.Linear(dim, self.num_heads * self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def transpose_for_scores(self, x):
        B, token_nums, _ = x.size()
        x = x.view(B, token_nums, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn = (q @ k.transpose(-2, -1))

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        res = self.norm1(x)
        res = self.attn(res)
        res = self.drop_path(res)
        x = x + res
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        res = self.norm2(x)
        res = self.mlp(res)
        res = self.drop_path(res)
        x = x + res
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer
    An OneFlow impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(flow.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(flow.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(flow.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in flow.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # position embedding
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = flow.cat((cls_token, x), dim=1)
        else:
            x = flow.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        # classification head
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training:
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
#     """ ViT weight initialization
#     * When called without n, head_bias, jax_impl args it will behave exactly the same
#       as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
#     * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         elif name.startswith('pre_logits'):
#             lecun_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         else:
#             if jax_impl:
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#             else:
#                 trunc_normal_(module.weight, std=.02)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#     elif jax_impl and isinstance(module, nn.Conv2d):
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(module.bias)
#         nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = flow.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def _create_vision_transformer(arch, pretrained=False, progress=True, **model_kwargs):
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@ModelCreator.register_model
def vit_tiny_patch16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Tiny-patch16-224 model.
    .. note::
        ViT-Tiny-patch16-224 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 224x224.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_tiny_patch16_224 = flowvision.models.vit_tiny_patch16_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs
    )
    model = _create_vision_transformer("vit_tiny_patch16_224", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_tiny_patch16_384(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Tiny-patch16-384 model.
    .. note::
        ViT-Tiny-patch16-384 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 384x384.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_tiny_patch16_384 = flowvision.models.vit_tiny_patch16_384(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=384,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs
    )
    model = _create_vision_transformer("vit_tiny_patch16_384", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_small_patch32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Small-patch32-224 model.
    .. note::
        ViT-Small-patch32-224 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 224x224.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_small_patch32_224 = flowvision.models.vit_small_patch32_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=224,
        patch_size=32,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    model = _create_vision_transformer("vit_small_patch32_224", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_small_patch32_384(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Small-patch32-384 model.
    .. note::
        ViT-Small-patch32-384 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 384x384.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_small_patch32_384 = flowvision.models.vit_small_patch32_384(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=384,
        patch_size=32,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    model = _create_vision_transformer("vit_tiny_patch16_384", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_small_patch16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Small-patch16-224 model.
    .. note::
        ViT-Small-patch16-224 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 224x224.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_small_patch16_224 = flowvision.models.vit_small_patch16_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    model = _create_vision_transformer("vit_small_patch16_224", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_small_patch16_384(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Small-patch16-384 model.
    .. note::
        ViT-Small-patch16-384 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 384x384.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_small_patch16_384 = flowvision.models.vit_small_patch16_384(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    model = _create_vision_transformer("vit_small_patch16_384", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_base_patch32_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Base-patch32-224 model.
    .. note::
        ViT-Base-patch32-224 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 224x224.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_base_patch32_224 = flowvision.models.vit_base_patch32_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_vision_transformer("vit_base_patch32_224", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_base_patch32_384(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Base-patch32-384 model.
    .. note::
        ViT-Base-patch32-384 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 384x384.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_base_patch32_384 = flowvision.models.vit_base_patch32_384(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_vision_transformer("vit_base_patch32_384", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_base_patch16_224(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Base-patch16-224 model.
    .. note::
        ViT-Base-patch16-224 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 224x224.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_base_patch16_224 = flowvision.models.vit_base_patch16_224(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, progress=progress, **model_kwargs)
    return model


@ModelCreator.register_model
def vit_base_patch16_384(pretrained=False, progress=True, **kwargs):
    """
    Constructs the ViT-Base-patch16-384 model.
    .. note::
        ViT-Base-patch16-384 model from `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/pdf/2010.11929.pdf>`_.
        The required input size of the model is 384x384.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> vit_base_patch16_384 = flowvision.models.vit_base_patch16_384(pretrained=False, progress=True)
    """
    model_kwargs = dict(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_vision_transformer("vit_base_patch16_384", pretrained=pretrained, progress=progress, **model_kwargs)
    return model
