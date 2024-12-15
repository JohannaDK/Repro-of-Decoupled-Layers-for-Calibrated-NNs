# Some code from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import Tuple, Union
from src.models.ResBlock import *
from src.models.WRN import _BlockGroup
from src.models.ViT import *
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel, ViTConfig
from data.tinyimagenet import *
from src.models.ResNet_new import *
from src.models.EfficientNet_new import *

WIDERESNET_WIDTH_WANG2023=10
WIDERESNET_WIDTH_MNIST=4
ResNet50_block_expansion = 4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class WRN2810VarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x


class WRN2810VarHeadMLP4(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*6)
        self.fc2 = nn.Linear(latent_dim*6, latent_dim*3)
        self.fc3 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.logsigmoid(self.fc3(x))
        return x


class WRN2810VarHeadMLP5(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*9)
        self.fc2 = nn.Linear(latent_dim*9, latent_dim*6)
        self.fc3 = nn.Linear(latent_dim*6, latent_dim*3)
        self.fc4 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.logsigmoid(self.fc4(x))
        return x


class WRN2810Head(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WRN2810HeadMLP4(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*6)
        self.fc2 = nn.Linear(latent_dim*6, latent_dim*3)
        self.fc3 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WRN2810HeadMLP5(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*9)
        self.fc2 = nn.Linear(latent_dim*9, latent_dim*6)
        self.fc3 = nn.Linear(latent_dim*6, latent_dim*3)
        self.fc4 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class WRN2810Body(nn.Module):
    """
    Adapted WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """
    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.num_input_channels = num_input_channels
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        #self.fc = nn.Linear(num_channels[3], latent_dim)
        self.num_channels = num_channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return out

class CNNVarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(84, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class CNNHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(84, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNBody(nn.Module):
    """
    CNN model
    Arguments:
        num_classes (int): number of output classes.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, num_classes) # skip last layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc3(x) # don't use last layer, instead MLP head is added here
        return x


class ViTVarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(768, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class ViTHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(768, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ViTBody(nn.Module):
    """
    ViT model
    Arguments:
        dataset (str): name of the dataset to be used.
        model_name_or_path (str): pretrained weights to be used
    """
    def __init__(self, dataset, model_name_or_path):
        super().__init__()
        if dataset == "CIFAR10":
            labels = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        elif dataset == "TINYIMAGENET":
            labels = get_tinyimagenet_labels_from_dataset(os.getcwd()+"/data/")
        else:
             raise Exception("Oops, this dataset cannot be combined with a ViT!")

        config = ViTConfig.from_pretrained(model_name_or_path)
        config.add_pooling_layer = False
        config.num_labels = len(labels)
        config.id2label = {str(i): c for i, c in enumerate(labels)}
        config.label2id = {c: str(i) for i, c in enumerate(labels)}
        self.vit = ViTModel.from_pretrained(
                        model_name_or_path,
                        config=config
                    )

    def forward(self, x):
        outputs = self.vit(x)
        sequence_output = outputs[0]

        return sequence_output[:, 0, :]


class ResNet50VarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(512*ResNet50_block_expansion, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class ResNet50Head(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(512*ResNet50_block_expansion, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNet50Body(nn.Module):
    """Adapted ResNet50 model in which the last linear layer is removed"""
    def __init__(self, block = Bottleneck , num_blocks = [3, 4, 6, 3], num_classes=10):
        super(ResNet50Body, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out

class EfficientNetB5Body(nn.Module):
    def __init__(self, cfg = {
        'num_blocks': [3, 4, 6, 6, 8, 10, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [32, 48, 80, 160, 224, 384, 640],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.4,
        'drop_connect_rate': 0.2,
        }, num_classes=10):
        super(EfficientNetB5Body, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        #self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        #out = self.linear(out)
        return out

class EfficientNetB5VarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(640, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class EfficientNetB5Head(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(640, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#---------------OLD EFFICIENTNET-----------------------

class EfficientNetBody_old(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks_args, global_params = get_efficientnetv2_params("efficientnetv2-m", num_classes=num_classes)
        self._blocks_args = blocks_args
        self._global_params = global_params
        # stem
        inc = 3
        outc = blocks_args[0].input_filters
        self._stem_conv = Conv2dAutoPadding(inc, outc, 3, 1) #MODIFIED FOR CIFAR10 to have stride=1
        self._bn0 = nn.BatchNorm2d(outc)
        # blocks
        self._blocks = nn.ModuleList([]) # BUG: [] -> nn.ModuleList([])
        for block_arg in self._blocks_args:
            block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
            self._blocks.append(block)
            if block_arg.num_repeat > 1:
                block_arg = block_arg._replace(input_filters=block_arg.output_filters, stride=1)
            for _ in range(block_arg.num_repeat - 1):
                block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
                self._blocks.append(block)
        # head
        inc = block_arg.output_filters
        outc = int(self._global_params.width_coefficient * 1280)
        self._head_conv = nn.Conv2d(inc, outc, 1, 1)
        self._bn1 = nn.BatchNorm2d(outc)
        # top
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate) # missing dropout
        #self._fc = nn.Linear(outc, self._global_params.num_classes)
        # activation
        self._swish = nn.SiLU()  # hasattr?


    def forward(self, inputs):
        x = self._swish(self._bn0(self._stem_conv(inputs)))

        for i, block in enumerate(self._blocks): # BUG: missing enumerate
            x = block(x)

        x = self._swish(self._bn1(self._head_conv(x)))

        x = self._avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        #x = self._fc(x)
        return x

class EfficientNetVarHead_old(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(1280, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class EfficientNetHead_old(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(1280, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x