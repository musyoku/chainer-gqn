import cupy
import numpy
import chainer
from chainer import configuration, function
from chainer.backends import cuda
from chainer.utils import type_check
from chainer.utils import conv


def cudnn_relu_forward(x):
    return cuda.cudnn.activation_forward(x,
                                         cuda.cuda.cudnn.CUDNN_ACTIVATION_RELU)


def relu_backward(x, gy):
    return gy * (x > 0).astype(x.dtype)


def cudnn_convolution_forward(x, W, b, out_size, pad=0, stride=0):
    oc = W.shape[0]
    batch_size = x.shape[0]
    oh, ow = out_size
    y = cuda.cupy.empty((batch_size, oc, oh, ow), dtype=x.dtype)

    pad = (pad, pad)
    stride = (stride, stride)
    cuda.cudnn.convolution_forward(
        x,
        W,
        b,
        y,
        pad,
        stride, (1, 1),
        1,
        auto_tune=configuration.config.autotune,
        tensor_core=configuration.config.use_cudnn_tensor_core)
    return y


def convolution_backward(x, W, gy, pad=0, stride=0, cover_all=False):
    xh, xw = x.shape[2:]
    gx = chainer.functions.deconvolution_2d(
        gy,
        W,
        stride=(stride, stride),
        pad=(pad, pad),
        outsize=(xh, xw),
        dilate=(1, 1),
        groups=1).data

    if (not (gy.flags.c_contiguous or gy.flags.f_contiguous)
            and 1 in gy.shape):
        gy = numpy.ascontiguousarray(gy)

    kh, kw = W.shape[2:]
    out_channels = gy.shape[1]
    in_channels = x.shape[1]

    gW = cuda.cupy.empty((out_channels, in_channels, kh, kw), dtype=W.dtype)
    deterministic = configuration.config.cudnn_deterministic
    auto_tune = configuration.config.autotune
    tensor_core = configuration.config.use_cudnn_tensor_core
    cuda.cudnn.convolution_backward_filter(
        x,
        gy,
        gW, (pad, pad), (stride, stride), (1, 1),
        1,
        deterministic=deterministic,
        auto_tune=auto_tune,
        tensor_core=tensor_core)

    gb = cupy.sum(gy, axis=(0, 2, 3))
    return gx, gW, gb


class RepresentationTowerFunction(function.Function):
    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError

    def forward_gpu(self, inputs):
        x = inputs[0]
        v = inputs[1]
        conv1_1_W, conv1_1_b = inputs[2], inputs[3]
        conv1_2_W, conv1_2_b = inputs[4], inputs[5]
        conv1_res_W, conv1_res_b = inputs[6], inputs[7]
        conv1_3_W, conv1_3_b = inputs[8], inputs[9]
        conv2_1_W, conv2_1_b = inputs[10], inputs[11]
        conv2_res_W, conv2_res_b = inputs[12], inputs[13]
        conv2_2_W, conv2_2_b = inputs[14], inputs[15]
        conv2_3_W, conv2_3_b = inputs[16], inputs[17]

        self.conv_inputs = []
        self.relu_inputs = []

        image_height = x.shape[2]
        image_width = x.shape[3]

        self.conv_inputs.append(x)
        relu_in = cudnn_convolution_forward(
            x,
            conv1_1_W,
            conv1_1_b, (image_height // 2, image_width // 2),
            pad=0,
            stride=2)
        self.relu_inputs.append(relu_in)
        resnet_in = cudnn_relu_forward(relu_in)

        self.conv_inputs.append(resnet_in)
        relu_in = cudnn_convolution_forward(
            resnet_in,
            conv1_2_W,
            conv1_2_b, (image_height // 2, image_width // 2),
            pad=1,
            stride=1)
        self.relu_inputs.append(relu_in)
        out = cudnn_relu_forward(relu_in)

        self.conv_inputs.append(resnet_in)
        relu_in = cudnn_convolution_forward(
            resnet_in,
            conv1_res_W,
            conv1_res_b, (image_height // 4, image_width // 4),
            pad=0,
            stride=2)
        self.relu_inputs.append(relu_in)
        residual = cudnn_relu_forward(relu_in)

        self.conv_inputs.append(out)
        relu_in = cudnn_convolution_forward(
            out,
            conv1_3_W,
            conv1_3_b, (image_height // 4, image_width // 4),
            pad=0,
            stride=2)
        self.relu_inputs.append(relu_in)
        out = cudnn_relu_forward(relu_in) + residual

        resnet_in = cupy.concatenate((out, v), axis=1)

        self.conv_inputs.append(resnet_in)
        relu_in = cudnn_convolution_forward(
            resnet_in,
            conv2_1_W,
            conv2_1_b, (image_height // 4, image_width // 4),
            pad=1,
            stride=1)
        self.relu_inputs.append(relu_in)
        out = cudnn_relu_forward(relu_in)

        self.conv_inputs.append(resnet_in)
        relu_in = cudnn_convolution_forward(
            resnet_in,
            conv2_res_W,
            conv2_res_b, (image_height // 4, image_width // 4),
            pad=1,
            stride=1)
        self.relu_inputs.append(relu_in)
        residual = cudnn_relu_forward(relu_in)

        self.conv_inputs.append(out)
        relu_in = cudnn_convolution_forward(
            out,
            conv2_2_W,
            conv2_2_b, (image_height // 4, image_width // 4),
            pad=1,
            stride=1)
        self.relu_inputs.append(relu_in)
        out = cudnn_relu_forward(relu_in) + residual

        self.conv_inputs.append(out)
        out = cudnn_convolution_forward(
            out,
            conv2_3_W,
            conv2_3_b, (image_height // 4, image_width // 4),
            pad=0,
            stride=1)

        return out,

    def backward_gpu(self, inputs, grad_outputs):
        x = inputs[0]
        v = inputs[1]
        conv1_1_W, conv1_1_b = inputs[2], inputs[3]
        conv1_2_W, conv1_2_b = inputs[4], inputs[5]
        conv1_res_W, conv1_res_b = inputs[6], inputs[7]
        conv1_3_W, conv1_3_b = inputs[8], inputs[9]
        conv2_1_W, conv2_1_b = inputs[10], inputs[11]
        conv2_res_W, conv2_res_b = inputs[12], inputs[13]
        conv2_2_W, conv2_2_b = inputs[14], inputs[15]
        conv2_3_W, conv2_3_b = inputs[16], inputs[17]

        gy = grad_outputs[0]
        ret = []

        conv_in = self.conv_inputs.pop(-1)
        gy_2_3, gW, gb = convolution_backward(
            conv_in, conv2_3_W, gy, pad=0, stride=1)
        ret.append(gb)
        ret.append(gW)

        relu_in = self.relu_inputs.pop(-1)
        gy_2_2 = relu_backward(relu_in, gy_2_3)

        conv_in = self.conv_inputs.pop(-1)
        gy_2_2, gW, gb = convolution_backward(
            conv_in, conv2_2_W, gy_2_2, pad=1, stride=1)
        ret.append(gb)
        ret.append(gW)

        relu_in = self.relu_inputs.pop(-1)
        gy_2_res = relu_backward(relu_in, gy_2_3)

        conv_in = self.conv_inputs.pop(-1)
        gy_2_res, gW, gb = convolution_backward(
            conv_in, conv2_res_W, gy_2_res, pad=1, stride=1)
        ret.append(gb)
        ret.append(gW)

        relu_in = self.relu_inputs.pop(-1)
        gy_2_1 = relu_backward(relu_in, gy_2_2)

        conv_in = self.conv_inputs.pop(-1)
        gy_2_1, gW, gb = convolution_backward(
            conv_in, conv2_1_W, gy_2_1, pad=1, stride=1)
        ret.append(gb)
        ret.append(gW)

        out_channels = conv1_3_W.shape[0]
        gy_concat = gy_2_res[:, :out_channels] + gy_2_1[:, :out_channels]
        gv = gy_2_res[:, out_channels:] + gy_2_1[:, out_channels:]

        relu_in = self.relu_inputs.pop(-1)
        gy_1_3 = relu_backward(relu_in, gy_concat)

        conv_in = self.conv_inputs.pop(-1)
        gy_1_3, gW, gb = convolution_backward(
            conv_in, conv1_3_W, gy_1_3, pad=0, stride=2)
        ret.append(gb)
        ret.append(gW)

        relu_in = self.relu_inputs.pop(-1)
        gy_1_res = relu_backward(relu_in, gy_concat)

        conv_in = self.conv_inputs.pop(-1)
        gy_1_res, gW, gb = convolution_backward(
            conv_in, conv1_res_W, gy_1_res, pad=0, stride=2)
        ret.append(gb)
        ret.append(gW)

        relu_in = self.relu_inputs.pop(-1)
        gy_1_2 = relu_backward(relu_in, gy_1_3)

        conv_in = self.conv_inputs.pop(-1)
        gy_1_2, gW, gb = convolution_backward(
            conv_in, conv1_2_W, gy_1_2, pad=1, stride=1)
        ret.append(gb)
        ret.append(gW)

        gy_residual = gy_1_res + gy_1_2

        relu_in = self.relu_inputs.pop(-1)
        gy_1_1 = relu_backward(relu_in, gy_residual)

        conv_in = self.conv_inputs.pop(-1)
        gx, gW, gb = convolution_backward(
            conv_in, conv1_1_W, gy_1_1, pad=0, stride=2)
        ret.append(gb)
        ret.append(gW)

        ret.append(gv)
        ret.append(gx)

        ret.reverse()
        return ret


class CoreFunction(function.Function):
    def check_type_forward(self, in_types):
        type_check._argname(in_types, ("gate_inputs", "prev_cg"))

        type_check.expect(in_types[0].dtype.kind == "f")
        type_check.expect(in_types[1].dtype.kind == "f")

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError

    def forward_gpu(self, inputs):
        gate_inputs, prev_c = inputs

        forget_gate_input, input_gate_input, tanh_input, output_gate_input = cupy.split(
            gate_inputs, 4, axis=1)

        kernel_input = "T forget_gate_input, T input_gate_input, T tanh_input, T output_gate_input, T prev_c"
        kernel_outputs = "T forget_gate, T input_gate, T tanh_gate, T output_gate, T tanh_next_c, T next_h, T next_c"

        kernel = "forget_gate = (tanh(forget_gate_input * 0.5f) + 1.0f) * 0.5f;"\
            "input_gate = (tanh(input_gate_input * 0.5f) + 1.0f) * 0.5f;"\
            "tanh_gate = tanh(tanh_input);"\
            "next_c = forget_gate * prev_c + input_gate * tanh_gate;"\
            "output_gate = (tanh(output_gate_input * 0.5f) + 1.0f) * 0.5f;"\
            "tanh_next_c = tanh(next_c);"\
            "next_h = output_gate * tanh(next_c)"

        (self.forget_gate, self.input_gate, self.tanh_gate, self.output_gate,
         self.tanh_next_c, self.next_h, self.next_c) = cuda.elementwise(
             kernel_input, kernel_outputs, kernel,
             'gqn_core_fwd')(forget_gate_input, input_gate_input, tanh_input,
                             output_gate_input, prev_c)

        return self.next_h, self.next_c

    def backward_gpu(self, inputs, grad_outputs):
        gate_inputs, prev_c = inputs

        forget_gate_input, input_gate_input, tanh_input, output_gate_input = cupy.split(
            gate_inputs, 4, axis=1)

        grad_next_h = grad_outputs[0]

        kernel_input = "T forget_gate, T input_gate, T tanh_gate, T output_gate, T tanh_next_c, T prev_c, T grad_next_h"
        kernel_outputs = "T grad_forget_gate_input, T grad_input_gate_input, T grad_tanh_input, T grad_output_gate_input, T grad_prev_c"

        kernel = "grad_output_gate_input = grad_next_h * tanh_next_c * output_gate * (1.0f - output_gate);"\
            "T grad_c = grad_next_h * output_gate * (1.0f - tanh_next_c * tanh_next_c);"\
            "grad_forget_gate_input = grad_c * prev_c * forget_gate * (1.0f - forget_gate);"\
            "grad_input_gate_input = grad_c * tanh_gate * input_gate * (1.0f - input_gate);"\
            "grad_tanh_input = grad_c * input_gate * (1.0f - tanh_gate * tanh_gate);"\
            "grad_prev_c = grad_c * forget_gate;"

        (grad_forget_gate_input, grad_input_gate_input, grad_tanh_input,
         grad_output_gate_input, grad_prev_c) = cuda.elementwise(
             kernel_input, kernel_outputs, kernel, 'gqn_core_fwd')(
                 self.forget_gate, self.input_gate, self.tanh_gate,
                 self.output_gate, self.tanh_next_c, prev_c, grad_next_h)

        grad_gate_inputs = cupy.concatenate(
            (grad_forget_gate_input, grad_input_gate_input, grad_tanh_input,
             grad_output_gate_input),
            axis=1)

        return grad_gate_inputs, grad_prev_c
