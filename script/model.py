from pretrainedmodels.models import bninception, nasnetalarge, resnet50
from torch import nn
from config import config


class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        probas = nn.Sigmoid()(output)
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


class AttendBNInception(nn.Module):
    def __init__(self, linear_module):
        super(AttendBNInception, self).__init__()
        self.bn = linear_module[0]
        self.drop = linear_module[1]
        self.linear1 = linear_module[2]
        self.linear2 = nn.Linear(in_features=1024, out_features=config.num_classes)
        self.attn = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.drop(x)
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        attn = self.attn(out2)
        x = out1.mul(attn)
        return x


class AttendResNet50(nn.Module):
    def __init__(self, linear_module):
        super(AttendResNet50, self).__init__()
        self.drop = linear_module[0]
        self.linear1 = linear_module[1]
        self.linear2 = nn.Linear(in_features=2048, out_features=config.num_classes)
        self.attn = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop(x)
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        attn = self.attn(out2)
        x = out1.mul(attn)
        return x


def f1_loss(output, target, epsilon=1e-7):
    probas = nn.Sigmoid()(output)
    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1 = f1.clamp(min=1e-15, max=1-1e-15)
    return 1 - f1.mean()


def get_bninception_model():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
    model.conv1_7x7_s2 = nn.Conv2d(
        in_channels=config.channels,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3)
    )
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=config.num_classes),
    )
    return model


def get_nasnet_model():
    model = nasnetalarge(pretrained='imagenet+background')
    model.conv0.conv = nn.Conv2d(
        in_channels=config.channels,
        out_channels=96,
        kernel_size=(3, 3),
        stride=(2, 2)
    )
    model.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
    model.last_linear = nn.Linear(in_features=4032, out_features=config.num_classes)
    return model


def get_resnet50_model():
    ''' fixed three channel model '''
    model = resnet50(pretrained="imagenet")
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    model.last_linear = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512, bias=True),
        nn.BatchNorm1d(num_features=512),
        nn.Linear(in_features=512, out_features=config.num_classes)
    )
    return model


def get_resnet50att_model():
    model = resnet50(pretrained="imagenet")
    model.conv1 = nn.Conv2d(
        in_channels=config.channels,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    model.last_linear = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=config.num_classes),
    )
    model.last_linear = AttendResNet50(model.last_linear)
    return model


def get_model(model_name):
    if model_name == "bninception":
        return get_bninception_model()
    if model_name == "nasnet_large":
        return get_nasnet_model()
    if model_name == "bninception_attn":
        model = get_bninception_model()
        model.last_linear = AttendBNInception(model.last_linear)
    if model_name == 'resnet50_attn':
        model = get_resnet50att_model()
    if model_name == 'resnet50':
        model = get_resnet50_model()
    return model


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    sample_input = Variable(torch.randn(2, config.channels, config.img_height, config.img_width))
    model = get_nasnet_model()
    sample_output = model(sample_input)
    assert list(sample_output.size()) == [2, config.num_classes], 'model output shape not correct'
