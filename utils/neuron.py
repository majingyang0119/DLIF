from abc import abstractmethod
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import logging
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate, base, lava_exchange
try:
    import cupy
    from spikingjelly.activation_based import cuda_kernel
    from spikingjelly.activation_based.cuda_kernel.auto_cuda import neuron_kernel as ac_neuron_kernel
    from spikingjelly.activation_based.cuda_kernel.auto_cuda import ss_neuron_kernel as ss_ac_neuron_kernel
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    cuda_kernel = None
    ac_neuron_kernel = None
    ss_ac_neuron_kernel = None

try:
    import triton
    from spikingjelly.activation_based import triton_kernel
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    triton = None
    triton_kernel = None


class DLIFNode(BaseNode):
    def __init__(
        self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
        v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False
    ):
        """
        * :ref:`API in English <DLIFNode.__init__-en>`

        .. _DLIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。在支持的情况下，使用 ``'cupy'`` 后端是速度最快的
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
        :type store_v_seq: bool

        Dendritic Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]


        * :ref:`中文API <DLIFNode.__init__-cn>`

        .. _DLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq
        )

        self.tau = tau
        self.decay_input = decay_input

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy', 'triton')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float, v_reset: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().single_step_forward(x)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None
                if x.dtype == torch.float:
                    dtype = 'float'
                elif x.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(
                    hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input
                ):
                    self.forward_kernel = ss_ac_neuron_kernel.LIFNodeFPKernel(
                        decay_input=self.decay_input, hard_reset=hard_reset, dtype=dtype
                    )

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input
                ):
                    self.backward_kernel = ss_ac_neuron_kernel.LIFNodeBPKernel(
                        decay_input=self.decay_input,
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype
                    )

                self.v_float_to_tensor(x)
                spike, v = ss_ac_neuron_kernel.LIFNodeATGF.apply(
                    x.flatten(0), self.v.flatten(0), self.v_threshold, 
                    self.v_reset, 1. / self.tau, self.forward_kernel, self.backward_kernel
                )
                spike = spike.reshape(x.shape)
                v = v.reshape(x.shape)
                self.v = v
                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(
                        x, self.v, self.v_threshold, self.tau
                    )
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(
                        x, self.v, self.v_threshold, self.tau
                    )
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(
                        x, self.v, self.v_threshold, self.v_reset, self.tau
                    )
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(
                        x, self.v, self.v_threshold, self.v_reset, self.tau
                    )
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(
                    hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input
                ):
                    self.forward_kernel = ac_neuron_kernel.LIFNodeFPTTKernel(
                        decay_input=self.decay_input, 
                        hard_reset=hard_reset, 
                        dtype=dtype
                    )
                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input
                ):
                    self.backward_kernel = ac_neuron_kernel.LIFNodeBPTTKernel(
                        decay_input=self.decay_input,
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype
                    )

                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = ac_neuron_kernel.LIFNodeATGF.apply(
                    x_seq.flatten(1), self.v.flatten(0),
                    self.v_threshold, self.v_reset, 1. / self.tau,
                    self.forward_kernel, self.backward_kernel
                )
                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
                return spike_seq
            elif self.backend == 'triton':
                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = triton_kernel.MultiStepLIFNodePTT.apply(
                    x_seq, self.v, self.decay_input, self.tau, self.v_threshold,
                    self.v_reset, self.detach_reset,
                    self.surrogate_function.triton_codes(),
                )
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])

            if self.backend == 'triton':
                spike_seq, v_seq = triton_kernel.MultiStepLIFNodePTT.apply(
                    x_seq, self.v, self.decay_input, self.tau, self.v_threshold,
                    self.v_reset, self.detach_reset,
                    self.surrogate_function.triton_codes(),
                )
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
                return spike_seq

            # torch & cupy backend:
            if self.v_reset is None:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau
                        )
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(
                            x_seq, self.v, self.v_threshold, self.tau
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau
                        )
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(
                            x_seq, self.v, self.v_threshold, self.tau
                        )
            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau
                        )
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau
                        )
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau
                        )
            return spike_seq

class SLTTNeuron(DLIFNode):
    def __init__(
        self,
        tau: float = 2.,
        decay_input: bool = False,
        v_threshold: float = 1.,
        v_reset: float = None,
        surrogate_function: Callable = None,
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
        **kwargs
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )


    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

class BPTTNeuron(DLIFNode):
    def __init__(
        self,
        tau: float = 2.,
        decay_input: bool = False,
        v_threshold: float = 1.,
        v_reset: float = None,
        surrogate_function: Callable = None,
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
        **kwargs
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )

class OnlineDLIFNode(DLIFNode):
    def __init__(
        self,
        tau: float = 2.,
        decay_input: bool = False,
        v_threshold: float = 1.,
        v_reset: float = None,
        surrogate_function: Callable = None,
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
        track_rate: bool = False,
        neuron_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking * (1 - 1. / self.tau) + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike
        

def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)

class _DLIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False):
        
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input

        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def forward(self, x: torch.Tensor):
        if self.cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32:
            # cupy is installed && eval mode && fp32
            device_id = x.get_device()
            if device_id < 0:
                return super().forward(x)

            # use cupy to accelerate
            if isinstance(self.v, float):
                v = torch.zeros_like(x)
                if self.v != 0.:
                    torch.fill_(v, self.v)
                self.v = v

            if self.v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            code = rf'''
                extern "C" __global__
                void LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward(
                const float * x, const float & v_threshold, {'const float & v_reset,' if hard_reset else ''} const float & tau,
                float * spike, float * v,
                const int & numel)
            '''

            code += r'''
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {
                        
            '''

            if self.decay_input:
                if hard_reset:
                    code += r'''
                                v[index] += (x[index] - (v[index] - v_reset)) / tau;
                            '''
                else:
                    code += r'''
                                v[index] += (x[index] - v[index]) / tau;
                    '''
            else:
                if hard_reset:
                    code += r'''
                                v[index] = x[index] + v[index] - (v[index] - v_reset) / tau;
                            '''
                else:
                    code += r'''
                                v[index] = x[index] + v[index] * (1.0f - 1.0f / tau);
                    '''

            code += rf'''
                        spike[index] = (float) (v[index] >= v_threshold);
                        {'v[index] = (1.0f - spike[index]) * v[index] + spike[index] * v_reset;' if hard_reset else 'v[index] -= spike[index] * v_threshold;'}
            '''

            code += r'''
                    }
                }
            '''
            if hasattr(self, 'cp_kernel'):
                if self.cp_kernel.code != code:
                    # replace codes
                    del self.cp_kernel
                    self.cp_kernel = cupy.RawKernel(code, f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)
            else:
                self.cp_kernel = cupy.RawKernel(code, f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)

            with cu_kernel_opt.DeviceEnvironment(device_id):
                numel = x.numel()
                threads = configure.cuda_threads
                blocks = cu_kernel_opt.cal_blocks(numel)
                cp_numel = cupy.asarray(numel)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                if hard_reset:
                    cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)
                cp_tau = cupy.asarray(self.tau, dtype=np.float32)
                spike = torch.zeros_like(x)
                if hard_reset:
                    x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel]
                else:
                    x, cp_v_threshold, cp_tau, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, cp_tau, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_tau, spike, self.v, cp_numel]

                self.cp_kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device_id,
                        *kernel_args
                    )
                )
                return spike
        else:
            return super().forward(x)
            
class MultiStepDLIFNode(_DLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)

    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()