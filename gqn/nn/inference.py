import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer import function_node


class InferenceCore(chainer.Chain):
    def __init__(
            self,
            h_channels=128,
            h_size=(16, 16),
            r_channels=256,
            r_size=(16, 16),
            u_channels=128,
            use_cuda_kernel=False,
            nobias_broadcast=True,
            weight_initializer=None,
    ):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        with self.init_scope():
            self.downsample_xu = nn.Convolution2D(
                None,
                3 + u_channels,
                ksize=4,
                stride=4,
                pad=0,
                initialW=weight_initializer)
            self.broadcast_v = nn.Deconvolution2D(
                7,
                7,
                ksize=h_size,
                pad=0,
                stride=h_size,
                initialW=weight_initializer,
                nobias=nobias_broadcast)
            if r_size[0] == 1:
                self.broadcast_r = nn.Deconvolution2D(
                    r_channels,
                    r_channels,
                    ksize=h_size,
                    pad=0,
                    stride=h_size,
                    initialW=weight_initializer,
                    nobias=nobias_broadcast)

            self.lstm = nn.Convolution2D(
                None,
                h_channels * 4,
                ksize=5,
                stride=1,
                pad=2,
                initialW=weight_initializer)

    def __call__(self, prev_hg, prev_he, prev_ce, x, v, r, u):
        xu = cf.concat((x, u), axis=1)
        xu = self.downsample_xu(xu)
        v = self.broadcast_v(v)
        if r.shape[2] == 1:
            r = self.broadcast_r(r)

        lstm_input = cf.concat((prev_he, prev_hg, xu, v, r), axis=1)
        gate_inputs = self.lstm(lstm_input)

        if self.use_cuda_kernel:
            next_h, next_c = CoreFunction()(gate_inputs, prev_ce)
        else:
            forget_gate_input, input_gate_input, tanh_input, output_gate_input = cf.split_axis(
                gate_inputs, 4, axis=1)

            forget_gate = cf.sigmoid(forget_gate_input)
            input_gate = cf.sigmoid(input_gate_input)
            next_c = forget_gate * prev_ce + input_gate * cf.tanh(tanh_input)
            output_gate = cf.sigmoid(output_gate_input)
            next_h = output_gate * cf.tanh(next_c)

        return next_h, next_c
