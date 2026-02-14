from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from utils.neuron import MultiStepDLIFNode
from module import *

class FusionBilinearHead(nn.Module):
    def __init__(self, dim, num_classes, bias=True):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Bilinear weight: [num_classes, dim, dim]
        self.weight_bilinear = nn.Parameter(
            torch.randn(num_classes, dim, dim) * 0.02
        )

        # Linear part: [num_classes, dim]
        self.weight_linear = nn.Parameter(
            torch.randn(num_classes, dim) * 0.02
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: [T, B, C]
        output: [T, B, num_classes]
        """

        T, B, C = x.shape

        # outer product x ⊗ x --> [T, B, C, C]
        outer = torch.einsum("tbc,tbd->tbcd", x, x)

        # bilinear output: [T, B, num_classes]
        y_bilinear = torch.einsum(
            "tbcd,kcd->tbk", outer, self.weight_bilinear
        )

        # linear output
        y_linear = torch.einsum(
            "tbc,kc->tbk", x, self.weight_linear
        )

        y = y_bilinear + y_linear
        
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)

        return y

class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "dlif":
            self.head_lif = MultiStepDLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.head = (
            FusionBilinearHead(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
