import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import base, functional
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
from typing import Optional, List, Tuple, Union
from typing import Callable
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair
from torch.nn import init

class MultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[T, batch_size, ...]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[T, batch_size, ...]``
        :rtype: Tensor
        """
        return functional.multi_step_forward(x_seq, super().forward)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: Tensor
        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: Tensor
        """
        return functional.seq_to_ann_forward(x_seq, super().forward)



class TLastMultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!")
    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[batch_size, ..., T]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[batch_size, ..., T]``
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward)
    
class TLastSeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        Please refer to :class:`spikingjelly.activation_based.functional.t_last_seq_to_ann_forward` .
        """
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!")


    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[batch_size, ..., T]
        :type x_seq: Tensor
        :return: y_seq, shape=[batch_size, ..., T]
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward) 

class StepModeContainer(nn.Sequential, base.StepModule):
    def __init__(self, stateful: bool, *args):
        super().__init__(*args)
        self.stateful = stateful
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by StepModeContainer!")
        self.step_mode = 's'


    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            if self.stateful:
                return functional.multi_step_forward(x, super().forward)
            else:
                return functional.seq_to_ann_forward(x, super().forward)
            
class Conv2d(nn.Conv2d, base.StepModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <Conv2d-en>`

        .. _Conv2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv2d`

        * :ref:`中文 API <Conv2d-cn>`

        .. _Conv2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv2d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x

class Conv2d_bilinear(Conv2d, base.StepModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 1, 
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            step_mode: str = 's',
            sparsity_level: float = 0.0
    ) -> None:
        """
        双线性卷积层，支持 SpikingJelly 的 step_mode。
        计算逻辑：对输入特征进行外积 (Channel wise outer product)，然后进行线性映射。
        """
        super(Conv2d_bilinear, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.step_mode = step_mode
        self.sparsity_level = sparsity_level

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels , in_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.create_mask()

        self.reset_parameters()

    def create_mask(self):
        mask = (torch.rand(self.out_channels, self.in_channels, self.in_channels) > self.sparsity_level).float()
        
        for i in range(self.out_channels):
            mask[i].fill_diagonal_(0)
            
        self.register_buffer('mask', mask)   

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        if hasattr(self, 'mask'):
            with torch.no_grad():
                self.weight.data.mul_(self.mask)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'step_mode={self.step_mode}')

    def _core_forward(self, input):
        qinput = torch.bmm(input.transpose(1,3).reshape(-1,self.in_channels).unsqueeze(-1), input.transpose(1,3).reshape(-1,self.in_channels).unsqueeze(-2)).reshape(-1,self.in_channels**2)
        masked_weight = (self.weight * self.mask).reshape(self.out_channels, -1)
        y = F.linear(qinput, masked_weight).reshape(input.size(0),input.size(2),input.size(3),self.out_channels).transpose(1,3).transpose(2,3)
        return y

    def forward(self, x: Tensor):
        # SpikingJelly 标准的 step_mode 处理逻辑
        if self.step_mode == 's':
            # 单步模式：输入 x 为 [N, C, H, W]
            x = self._core_forward(x)

        elif self.step_mode == 'm':
            # 多步模式：输入 x 为 [T, N, C, H, W]
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            
            # 使用 seq_to_ann_forward 将 T 和 N 合并后并行计算，速度最快
            x = functional.seq_to_ann_forward(x, self._core_forward)
            
        return x
    

class Conv2d_bilinear_v(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        sparsity_level=0.9,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sparsity_level = sparsity_level

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, in_channels) * 0.02
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        mask = (torch.rand(out_channels, in_channels, in_channels) > sparsity_level).float()
        for i in range(out_channels):
            mask[i].fill_diagonal_(0)
        self.register_buffer("mask", mask)

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

    def forward(self, x):
        N, C, H, W = x.shape

        patches = self.unfold(x)               
        # (N, C*kh*kw, L)

        patches = patches.transpose(1, 2)      
        # (N, L, C*kh*kw)

        patches = patches.reshape(N, -1, C, self.kernel_size[0], self.kernel_size[1])
        patches = patches.mean(dim=[3, 4])     
        # (N, L, C)

        # Bilinear outer product
        outer = torch.einsum('nlc,nld->nlcd', patches, patches)
        outer = outer.reshape(N, -1, C*C)      

        w = (self.weight * self.mask).reshape(self.out_channels, -1)  

        y = torch.einsum('nlc,oc->nlo', outer, w)  

        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)

        H_out = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1

        y = y.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)
        return y
