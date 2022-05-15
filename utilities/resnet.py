import torch
import torch.nn as nn
from lambda_layer import LambdaLayer

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "lambda_resnet50",
    "lambda_resnet38",
    "lambda_resnet26"
]


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        size,
        enable_lambda=False,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        #self.residual = residual
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = None
        #if self.residual:
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #if self.residual and self.downsample is not None:
        if self.downsample is not None:
            identity = self.downsample(x)

        #if self.residual:
        out += identity

        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        size,
        enable_lambda=False,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) 
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        if enable_lambda:
            self.conv2 = LambdaLayer(width, m=size, stride=stride)
        else:
            self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        enable_lambda,
        zero_init_residual=False,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
       
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],size=32,enable_lambda=enable_lambda)
        self.layer2 = self._make_layer(block, 128, layers[1],size=16,enable_lambda=enable_lambda,stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2],size=8,enable_lambda=enable_lambda,stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3],size=4,enable_lambda=enable_lambda,stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes,blocks,size,enable_lambda=False,stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                size,
                enable_lambda,
                stride,
                downsample,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    size,
                    enable_lambda,
                    stride=1,
                    downsample=None,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    arch,
    block,
    layers,
    num_classes,
    enable_lambda,
    pretrained,
    progress,
    **kwargs
):
    model = ResNet(block, layers, num_classes,enable_lambda,**kwargs)
    return model


def resnet18(
    num_classes, enable_lambda=False,pretrained=False, progress=True, **kwargs
):
    
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )


def resnet34(num_classes,enable_lambda=False,pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )


def resnet50(
    num_classes,enable_lambda=False,pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )

def lambda_resnet26(
    num_classes,enable_lambda=True,pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "lambda_resnet26",
        Bottleneck,
        [2, 2, 2, 2],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )
    
def lambda_resnet38(
    num_classes,enable_lambda=True,pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "lambda_resnet38",
        Bottleneck,
        [2, 3, 5, 2],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )

def lambda_resnet50(
    num_classes,enable_lambda=True,pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "lambda_resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        enable_lambda,
        pretrained,
        progress,
        **kwargs
    )


def resnet101(num_classes,enable_lambda=False,pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], num_classes,enable_lambda,pretrained, progress, **kwargs
    )


def resnet152(num_classes,enable_lambda=False,pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3],num_classes, enable_lambda,pretrained,progress, **kwargs
    )