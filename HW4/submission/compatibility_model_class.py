import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
class SiameseNetv3(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetv3, self).__init__()
        self.feat_extractor = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.feat_extractor.classifier = nn.Identity()
        self.fc = nn.Linear(1280*2,512)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x, y):
        x, y = self.feat_extractor(x), self.feat_extractor(y)
        x = torch.cat([x,y],1)
        print(x.shape)
        return F.sigmoid(self.classifier(self.dropout(nn.ReLU()(self.fc(x)))))

