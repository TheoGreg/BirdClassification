import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

nclasses = 20 

'''
6 Models developed here : 5 of them are used in main.py and evaluate.py to take part in Reliable Majority Voting.
'''

class TG_Resnet(nn.Module):
    """Class for custom resnet152 torch model,  pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_Resnet, self).__init__()
        
        model_pretrained = models.resnet152(pretrained=True)
        
        n_inputs = model_pretrained.fc.in_features
        model_pretrained.fc = nn.Sequential(
                      nn.Linear(n_inputs, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained.fc.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model.fc(x)
        return x
    
    
class TG_VGG(nn.Module):
    """Class for custom VGG16 torch model, pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_VGG, self).__init__()
        
        model_pretrained = models.vgg16(pretrained = True)
        
        for param in model_pretrained.parameters():
            param.requires_grad = False
                
        n_inputs = model_pretrained.classifier[0].in_features
        model_pretrained.classifier = nn.Sequential(
                      nn.Linear(n_inputs, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained.classifier.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

class TG_Densenet(nn.Module):
    """Class for custom DenseNet torch model, pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_Densenet, self).__init__()
        
        model_pretrained = models.densenet161(pretrained=True)

        model_pretrained.classifier = nn.Sequential(
                      nn.Linear(2208, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained.classifier.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.model.classifier(out)
        return out


class TG_Resnext(nn.Module):
    """Class for custom resnext50_32x4d torch model, pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_Resnext, self).__init__()
        
        model_pretrained = models.resnext50_32x4d(pretrained=True)

        n_inputs = model_pretrained.fc.in_features
        model_pretrained.fc = nn.Sequential(
                      nn.Linear(n_inputs, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained.fc.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model.fc(x)
        return x


class TG_Alexnet(nn.Module):
    """Class for custom AlexNet torch model, pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_Alexnet, self).__init__()
        
        model_pretrained = models.alexnet(pretrained=True)
        
        model_pretrained.classifier = nn.Sequential(
                      nn.Linear(9216, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained.classifier.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

class TG_Efficientnet(nn.Module):
    """Class for custom Efficientnet-B0 torch model, pretrained on ImageNet."""

    def __init__(self, class_num=nclasses, droprate=0.4):

        super(TG_Efficientnet, self).__init__()
        
        model_pretrained = EfficientNet.from_pretrained('efficientnet-b0')

        n_inputs = model_pretrained._fc.in_features

        model_pretrained._fc = nn.Sequential(
                      nn.Linear(n_inputs, 4096), 
                      nn.ReLU(), 
                      nn.Dropout(droprate),
                      nn.Linear(4096, nclasses),                   
                      nn.LogSoftmax(dim=1))
        
        model_pretrained._fc.requires_grad = True
        
        self.model = model_pretrained

    def forward(self, x):
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        return x
