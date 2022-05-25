from numpy import s_
import torch
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU, Sequential, AvgPool2d, Linear, Dropout


class GoogleNet(Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(GoogleNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = ConvolutionBlock(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvolutionBlock(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvolutionBlock(
            in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(
            in_channels=192, num_1x1=64, num_3x3_reduce=96, num_3x3=128, num_5x5_reduce=16, num_5x5=32, pool_proj=32)
        self.inception3b = InceptionBlock(
            in_channels=256, num_1x1=128, num_3x3_reduce=128, num_3x3=192, num_5x5_reduce=32, num_5x5=96, pool_proj=64)
        self.maxpool3 = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(
            in_channels=480, num_1x1=192, num_3x3_reduce=96, num_3x3=208, num_5x5_reduce=16, num_5x5=48, pool_proj=64)
        self.inception4b = InceptionBlock(
            in_channels=512, num_1x1=160, num_3x3_reduce=112, num_3x3=224, num_5x5_reduce=24, num_5x5=64, pool_proj=64)
        self.inception4c = InceptionBlock(
            in_channels=512, num_1x1=128, num_3x3_reduce=128, num_3x3=256, num_5x5_reduce=24, num_5x5=64, pool_proj=64)
        self.inception4d = InceptionBlock(
            in_channels=512, num_1x1=112, num_3x3_reduce=144, num_3x3=288, num_5x5_reduce=32, num_5x5=64, pool_proj=64)
        self.inception4e = InceptionBlock(
            in_channels=528, num_1x1=256, num_3x3_reduce=160, num_3x3=320, num_5x5_reduce=32, num_5x5=128, pool_proj=128)
        self.maxpool4 = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(
            in_channels=832, num_1x1=256, num_3x3_reduce=160, num_3x3=320, num_5x5_reduce=32, num_5x5=128, pool_proj=128)
        self.inception5b = InceptionBlock(
            in_channels=832, num_1x1=384, num_3x3_reduce=192, num_3x3=384, num_5x5_reduce=48, num_5x5=128, pool_proj=128)
        self.avgpool = AvgPool2d(kernel_size=7, stride=1)

        self.dropout = Dropout(p=0.4)
        self.fc1 = Linear(in_features=1024, out_features=num_classes)

        if self.aux_logits:
            self.aux4a = AuxiliaryClassifier(512, num_classes)
            self.aux4d = AuxiliaryClassifier(528, num_classes)
        else:
            self.aux4a = self.aux4d = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)

        out = self.inception4a(out)
        if self.aux_logits:
            aux1 = self.aux4a(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        if self.aux_logits:
            aux2 = self.aux4d(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)

        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc1(out)

        if self.aux_logits:
            return out, aux1, aux2
        else:
            return out


class InceptionBlock(Module):
    def __init__(self, in_channels, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvolutionBlock(
            in_channels, num_1x1, kernel_size=1, stride=1, padding=0)
        self.branch2 = Sequential(ConvolutionBlock(in_channels, num_3x3_reduce, kernel_size=1, stride=1,
                                  padding=0), ConvolutionBlock(num_3x3_reduce, num_3x3, kernel_size=3, padding=1, stride=1))
        self.branch3 = Sequential(ConvolutionBlock(in_channels, num_5x5_reduce, kernel_size=1, stride=1,
                                  padding=0), ConvolutionBlock(num_5x5_reduce, num_5x5, kernel_size=5, stride=1, padding=2))
        self.branch4 = Sequential(MaxPool2d(kernel_size=3, padding=1, stride=1), ConvolutionBlock(
            in_channels, pool_proj, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class AuxiliaryClassifier(Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()

        self.pool = AvgPool2d(kernel_size=5, stride=3)

        self.conv = ConvolutionBlock(
            in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()

        self.fc1 = Linear(in_features=2048, out_features=1024)
        self.dropout = Dropout(p=0.7)
        self.fc2 = Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        out = self.pool(x)

        out = self.conv(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out


class ConvolutionBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()

        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, padding=padding, stride=stride)

        self.relu = ReLU()

    def forward(self, x):
        out = self.conv(x)

        out = self.relu(out)

        return out


if __name__ == "__main__":
    google_net = GoogleNet(aux_logits=True)
    x = torch.randn(1, 3, 224, 224)
    print(google_net(x)[0].shape)
