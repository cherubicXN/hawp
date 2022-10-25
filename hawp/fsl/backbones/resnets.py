import torch
import torch.nn as nn
import torchvision

class ResNets(nn.Module):
    RESNET_TEMPLATES = {
        'resnet18':torchvision.models.resnet18,
        'resnet34':torchvision.models.resnet34,
        'resnet50':torchvision.models.resnet50,
        'resnet101':torchvision.models.resnet101,
    }
    def __init__(self,basenet, head, num_class, pretrain=True,):
        super(ResNets, self).__init__()
        assert basenet in ResNets.RESNET_TEMPLATES

        basenet_fn = ResNets.RESNET_TEMPLATES.get(basenet)

        model = basenet_fn(pretrain)
        
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.pixel_shuffle = nn.PixelShuffle(4)
        self.hafm_predictor = head(128,num_class)
        # self.hafm_predictor = nn.Sequential(nn.Conv2d(2048,512,3,1,1),nn.ReLU(True),nn.Conv2d(512,5,1,1,0))
    def forward(self, images):
        x = self.conv1(images)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pixel_shuffle(x)
        
        return [self.hafm_predictor(x)], x

if __name__ == "__main__":
    model = ResNets('resnet50')

    inp = torch.zeros((1,3,512,512))

    model(inp)


