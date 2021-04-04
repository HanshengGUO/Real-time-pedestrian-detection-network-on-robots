# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/13
# =====================
import torch
import torch.nn as nn

cfg1 = [22, 'M', 45, 'M', 90, 45, 90, 'M', 179, 90, 179]
cfg2 = ['M', 358, 179, 358, 179, 358]
cfg3 = ['M', 717, 358, 717, 358, 717]

def make_layers(cfg, in_channels=3, batch_norm=True, flag=True):
    """
    从配置参数中构建网络
    :param cfg:  参数配置
    :param in_channels: 输入通道数,RGB彩图为3, 灰度图为1
    :param batch_norm:  是否使用批正则化
    :return:
    """
    layers = []
    # flag = True             # 用于变换卷积核大小,(True选后面的,False选前面的)
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=v,
                                    kernel_size=(1, 3)[flag],
                                    stride=1,
                                    padding=(0, 1)[flag],
                                    bias=False))
            if batch_norm:
                bn = nn.BatchNorm2d(v)
                layers.append(bn)
            in_channels = v

            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        flag = not flag

    return nn.Sequential(*layers)


class SlimDarknet19(nn.Module):
    """
    Darknet19 模型
    """

    def __init__(self, in_channels=3, batch_norm=True, pretrained=False):
        """
        模型结构初始化
        :param in_channels: 输入数据的通道数  (input pic`s channel.)
        :param batch_norm:  是否使用正则化    (use batch_norm, True or False;True by default.)
        :param pretrained:  是否导入预训练参数 (use the pretrained weight)
        """
        super(SlimDarknet19, self).__init__()
        # 调用make_layers 方法搭建网络
        # (build the network)
        self.block1 = make_layers(cfg1, in_channels=in_channels, batch_norm=batch_norm, flag=True)
        self.block2 = make_layers(cfg2, in_channels=cfg1[-1], batch_norm=batch_norm, flag=False)
        self.block3 = make_layers(cfg3, in_channels=cfg2[-1], batch_norm=batch_norm, flag=False)
        # 导入预训练模型或初始化
        if pretrained:
            self.load_weight()
        else:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        return [feature1, feature2, feature3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_stages(self, stage):
        #############################
        # finetune剪枝模型只需要freeze bn,
        # 不需要freeze stage 1
        #############################
        if stage >= 0:
            self.block1[1].eval()
            for m in [self.block1[0], self.block1[1]]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, stage + 1):
            block = getattr(self, 'block{}'.format(i+1))
            block.eval()
            for param in m.parameters():
                param.requires_grad = False 

    def load_weight(self):
        # VOC-Slimdarknet19
        weight_file = '/data/FCOS-PyTorch-37.2AP/weight/darknet19_prune30_lasso5e_1.pth'
        # ImageNet-SlimDarknet19
        # weight_file = '/data/FCOS-PyTorch-37.2AP/imagenet_darknet19_prune30.pth'
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file).values()):
            dic[now_keys] = values
        self.load_state_dict(dic)


if __name__ == "__main__":
    darknet = SlimDarknet19()
    x = torch.randn((2, 3, 320, 480))
    y = darknet(x)
    for i in range(3):
        print(y[i].shape)
