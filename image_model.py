import torch
from torchvision import models


class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        input = x
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        grid_hidden = self.layer4(x)

        grid_hidden = grid_hidden.view(grid_hidden.size(0), grid_hidden.size(1), -1)
        grid_hidden = grid_hidden.permute((0, 2, 1))

        return grid_hidden


def init_net():
    model = models.resnet50(pretrained=True)
    model.__class__ = ModifiedResnet
    model.eval()
    return model
