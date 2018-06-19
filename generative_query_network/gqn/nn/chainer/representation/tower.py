import chainer
import chainer.functions as F
import chainer.links as L

from ...base.representation import RepresentationNetwork


class Links(chainer.Chain):
    def __init__(self):
        self.conv1_1 = L.Convolution2D(None, 256, ksize=2, pad=0, stride=2)
        self.conv1_2 = L.Convolution2D(None, 128, ksize=3, pad=1, stride=1)
        self.conv1_3 = L.Convolution2D(None, 256, ksize=2, pad=0, stride=2)
        self.conv2_1 = L.Convolution2D(None, 128, ksize=3, pad=1, stride=1)
        self.conv2_2 = L.Convolution2D(None, 256, ksize=3, pad=1, stride=1)
        self.conv2_3 = L.Convolution2D(None, 256, ksize=1, pad=0, stride=1)


class TowerRepresentationNetwork(RepresentationNetwork):
    def __init__(self):
        self.nn = Links()

    def compute_r(self, x, v):
        output = x
        output = F.relu(self.nn.conv1_1(output))
        output = F.relu(self.nn.conv1_2(output))
        output = F.relu(self.nn.conv1_3(output)) + output
        v = F.broadcast_to(v, shape=output.shape[:3] + v.shape[2])
        output = F.concat((output, v), axis=3)
        output = F.relu(self.nn.conv2_1(output))
        output = F.relu(self.nn.conv2_2(output)) + output
        output = F.relu(self.nn.conv2_3(output))
        return output
