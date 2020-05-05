import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
# from torch.autograd import Variable

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    high = 0.1
    low = 0.0
    alpha = 1
    max_iter = 10000.0

    # def __init__(self, high_value=1.0):
    #     super(AdversarialLayer, self).__init__()
    #     self.iter_num = 0
    #     self.alpha = 10
    #     self.low = 0.0
    #     self.high = high_value
    #     self.max_iter = 10000.0

    @staticmethod
    def forward(ctx, input):
        AdversarialLayer.iter_num += 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        coeff = np.float(2.0 * (AdversarialLayer.high - AdversarialLayer.low) / (1.0 + np.exp(-AdversarialLayer.alpha*AdversarialLayer.iter_num / AdversarialLayer.max_iter)) - (AdversarialLayer.high - AdversarialLayer.low) + AdversarialLayer.low) # 梯度求反的系数。最大为1， 最小为接近于0，系数绝对值随着迭代次数增加而减小。
        return -coeff * gradOutput

class SilenceLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        return 0 * gradOutput


# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=2):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_alexnet.classifier[6].in_features, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        else:
            self.fc = nn.Linear(model_alexnet.classifier[6].in_features, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        self.__in_features = bottleneck_dim
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152} 
class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=2):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=False)
    self.conv0 = nn.Conv2d(1, 3, kernel_size=7, stride=1, padding=3, dilation=1)
    self.conv1 = model_resnet.conv1 # 18 * 30
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool # 9 * 15
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    # self.avgpool = model_resnet.avgpool
    self.avgpool = nn.AvgPool2d(kernel_size=[9, 15], stride=1, padding=0, ceil_mode=False)
    self.feature_layers = nn.Sequential(self.conv0, self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool) # b * 2048 * 1 * 1

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    self.bottleneck_0 = nn.Linear(model_resnet.fc.in_features + 2, bottleneck_dim)
    self.bottleneck = nn.Linear(bottleneck_dim, bottleneck_dim)
    self.__in_features = bottleneck_dim

  def forward(self, x, pose):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    x = torch.cat([x, pose], 1)
    x = self.bottleneck_0(x)
    f = self.bottleneck(x)
    return f

  def output_num(self):
    return self.__in_features

class followingExtractor(nn.Module):
    def __init__(self, bottleneck_dim, class_num=2):
        super(followingExtractor, self).__init__()
        self.blocks = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.Linear(bottleneck_dim // 2, class_num)
        )

    def forward(self, x):
        return self.blocks(x)

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer1.bias.data.fill_(0)
    self.ad_layer2.bias.data.fill_(0)
    self.ad_layer3.bias.data.fill_(0)
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.ad_layer3(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class SmallAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(SmallAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 64)
    self.ad_layer2 = nn.Linear(64, 1)
    self.relu1 = nn.LeakyReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class LittleAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(LittleAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

