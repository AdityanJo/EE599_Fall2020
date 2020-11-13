from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
model = resnet50(pretrained=True)

class SusNetv2(nn.Module):
    # v1 is buggy with tensorboard :(
    def __init__(self, num_classes=10):
        super(SusNetv2, self).__init__()
        self.num_classes = num_classes
        self.preprocess = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch2=nn.MaxPool2d(2)

        self.abstractor_lvl_1_1_left = self.build_left_block(64, 64)
        self.abstractor_lvl_1_1_right = self.build_right_block(64, 64)
        self.squeeze_lvl_1_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_1_2_left = self.build_left_block(128, 128)
        self.abstractor_lvl_1_2_right = self.build_right_block(128, 128)
        self.squeeze_lvl_1_2 = nn.Conv2d(384, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_1_3_left = self.build_left_block(128, 128)
        self.abstractor_lvl_1_3_right = self.build_right_block(128, 256)
        self.squeeze_lvl_1_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        self.shrink_lvl_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.abstractor_lvl_2_1_left = self.build_left_block(128, 128)
        self.abstractor_lvl_2_1_right = self.build_right_block(128, 256)
        self.squeeze_lvl_2_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_2_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_2_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_3_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_3_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_4_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_4_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_4 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.shrink_lvl_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.abstractor_lvl_3_1_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_1_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_1 = nn.Conv2d(640, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_2_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_2_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_2 = nn.Conv2d(640, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_3_3_left = self.build_left_block(128, 128)
        self.abstractor_lvl_3_3_right = self.build_right_block(128, 128)
        self.squeeze_lvl_3_3 = nn.Conv2d(384, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_4_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_4_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_4 = nn.Conv2d(640, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_5_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_5_right = self.build_right_block(256, 256)
        self.squeeze_lvl_3_5 = nn.Conv2d(768, 512, kernel_size=1, stride=1)

        self.abstractor_lvl_3_6_left = self.build_left_block(512, 256)
        self.abstractor_lvl_3_6_right = self.build_right_block(512, 256)
        self.squeeze_lvl_3_6 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)

        self.shrink_lvl_3 = nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1)


        self.classifier=nn.Linear(2048,self.num_classes)

    def build_left_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features//2),
            nn.ReLU(),
            nn.Conv2d(out_features//2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    def build_right_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features*2),
            nn.ReLU(),
            nn.Conv2d(out_features*2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    def forward(self, x):

        x = self.preprocess(x)
        x1 = self.stem_branch1(x)
        x2 = self.stem_branch2(x)
        x = torch.cat([x1, x2], axis=1)

        x1 = self.abstractor_lvl_1_1_left(x)
        x2 = self.abstractor_lvl_1_1_right(x)
        x = self.squeeze_lvl_1_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_1_2_left(x)
        x2 = self.abstractor_lvl_1_2_right(x)
        x = self.squeeze_lvl_1_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_1_3_left(x)
        x2 = self.abstractor_lvl_1_3_right(x)
        x = self.squeeze_lvl_1_3(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_1(x)

        x1 = self.abstractor_lvl_2_1_left(x)
        x2 = self.abstractor_lvl_2_1_right(x)
        x = self.squeeze_lvl_2_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_2_left(x)
        x2 = self.abstractor_lvl_2_2_right(x)
        x = self.squeeze_lvl_2_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_3_left(x)
        x2 = self.abstractor_lvl_2_3_right(x)
        x = self.squeeze_lvl_2_3(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_4_left(x)
        x2 = self.abstractor_lvl_2_4_right(x)
        x = self.squeeze_lvl_2_4(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_2(x)

        x1 = self.abstractor_lvl_3_1_left(x)
        x2 = self.abstractor_lvl_3_1_right(x)
        x = self.squeeze_lvl_3_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_2_left(x)
        x2 = self.abstractor_lvl_3_2_right(x)
        x = self.squeeze_lvl_3_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_3_left(x)
        x2 = self.abstractor_lvl_3_3_right(x)
        x = self.squeeze_lvl_3_3(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_4_left(x)
        x2 = self.abstractor_lvl_3_4_right(x)
        x = self.squeeze_lvl_3_4(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_5_left(x)
        x2 = self.abstractor_lvl_3_5_right(x)
        x = self.squeeze_lvl_3_5(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_6_left(x)
        x2 = self.abstractor_lvl_3_6_right(x)
        x = self.squeeze_lvl_3_6(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_3(x)
        x=F.adaptive_avg_pool2d(x, (1,1))
        x=nn.Flatten()(x)
        x=self.classifier(x)
        return x

class SusNet(nn.Module):
    def __init__(self, device, num_classes=10):
        super(SusNet, self).__init__()
        self.num_classes=num_classes
        self.preprocess=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch1=nn.Sequential(
            nn.Conv2d(32,16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch2=nn.MaxPool2d(2)

        self.blocks_lvl_one=[DenseFeatureBlock(i,j).to(device) for i,j in [(64,128),(128,64),(64,128)]]
        self.squeeze_blocks_lvl_one=[nn.Conv2d(i,j,kernel_size=1, stride=1).to(device) for i,j in [(320,128),(256,64),(320,128)]]

        self.blocks_lvl_two=[DenseFeatureBlock(i,j).to(device) for i,j in [(128,64),(64,128),(128,256),(256,512)]]
        self.squeeze_blocks_lvl_two=[nn.Conv2d(i,j,kernel_size=1, stride=1).to(device) for i,j in [(256,64),(320,128),(640,256),(1280,512)]]

        self.final_block=nn.Sequential(
            nn.Conv2d(512, 704, kernel_size=3, stride=1),
            nn.BatchNorm2d(704),
            nn.ReLU(),
        )
        self.classifier=nn.Linear(704,self.num_classes)

    def forward(self,x):
        x=self.preprocess(x)
        x1=self.stem_branch1(x)
        x2=self.stem_branch2(x)
        x=torch.cat([x1,x2],1)

        for i in range(0,len(self.blocks_lvl_one)):
            x=self.blocks_lvl_one[i](x)
            x=self.squeeze_blocks_lvl_one[i](x)
        x=nn.AvgPool2d(2, stride=2)(x)

        for i in range(0,len(self.blocks_lvl_two)):
            x=self.blocks_lvl_two[i](x)
            print(x.shape)
            x=self.squeeze_blocks_lvl_two[i](x)
        x=nn.AvgPool2d(2, stride=2)(x)
        x=self.final_block(x)
        # print(x.shape)
        x=F.adaptive_avg_pool2d(x, (1,1))
        x=nn.Flatten()(x)
        x=self.classifier(x)
        # print(x.shape)
        return x

class DenseFeatureBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseFeatureBlock, self).__init__()
        self.branch1=nn.Sequential(
            nn.Conv2d(in_features, out_features//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features//2),
            nn.ReLU(),
            nn.Conv2d(out_features//2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_features, out_features*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features*2),
            nn.ReLU(),
            nn.Conv2d(out_features*2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    def forward(self, x):
        x1=self.branch1(x)
        x2=self.branch2(x)
        return torch.cat([x,x1,x2],1)

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

class SiameseNetv2(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetv2, self).__init__()
        self.feat_extractor = torchvision.models.resnet18(pretrained=pretrained)
        self.feat_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(512*2, 1)
    def forward(self, x, y):
        x, y = self.feat_extractor(x), self.feat_extractor(y)
        x = torch.cat([x,y],1)
        return F.sigmoid(self.classifier(x))

class SiameseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNet, self).__init__()
        self.feat_extractor = torchvision.models.resnet18(pretrained=pretrained)
        self.feat_extractor.fc = nn.Identity()
    def forward(self, x, y):
        return self.feat_extractor(x), self.feat_extractor(y)

class CompatNetv3(nn.Module):
    def __init__(self, pretrained=True):
        super(CompatNetv3, self).__init__()
        self.feat_extractor = torchvision.models.resnet50(pretrained=pretrained)
        self.feat_extractor.fc = nn.Linear(self.feat_extractor.fc.in_features, 1)

    def forward(self, x, y):
        x = torch.cat([x,y])
        x=self.feat_extractor(x)
        return F.sigmoid(x)

class CompatNetv2(nn.Module):
    def __init__(self, pretrained=True):
        super(CompatNetv2, self).__init__()
        self.branch1=torchvision.models.resnet50(pretrained=pretrained)
        self.branch1.fc=nn.Linear(self.branch1.fc.in_features,512)
        self.branch1_bn=nn.BatchNorm1d(512)

        self.branch2=torchvision.models.resnet50(pretrained=pretrained)
        self.branch2.fc=nn.Linear(self.branch2.fc.in_features,1)

        self.funnel = nn.Sequential(
                    nn.Linear(2*512, 64),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                )
        self.classifier=nn.Linear(64,1)

    def forward(self, x, y):
        x_1 = x
        x_2 = y

        x_1 = self.branch1(x_1)
        x_1 = self.branch1_bn(x_1)
        x_1 = nn.ReLU()(x_1)

        x_2 = self.branch1(x_2)
        x_2 = self.branch1_bn(x_2)
        x_2 = nn.ReLU()(x_2)

        x = torch.cat([x_1, x_2], axis=-1)
        x = self.funnel(x)
        x = self.classifier(x)
        return F.sigmoid(x)

class CompatNet(torchvision.models.resnet.ResNet):
    def __init__(self, pretrained=True):
        super(CompatNet, self).__init__(torchvision.models.resnet.Bottleneck,[3,4,6,3])
        if pretrained:
            self.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
        self.funnel1 =nn.Sequential(
                    nn.Linear(self.fc.in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU()
                )
        self.funnel2=nn.Sequential(
                    nn.Linear(self.fc.in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU()
                )
        self.funnel = nn.Sequential(
                    nn.Linear(2*512, 64),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                )
        self.classifier=nn.Linear(64,1)
    def _forward_impl(self, x):
        x_1 = x[0]
        x_2 = x[1]
        # print(x_2.shape)
        x = x_1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_1 = torch.flatten(x, 1)

        x = x_2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_2 = torch.flatten(x, 1)
        x_1 = self.funnel1(x_1)
        x_2 = self.funnel2(x_2)
        x = torch.cat([x_1,x_2], axis=-1)
        x = self.funnel(x)
        x=self.classifier(x)
        return F.sigmoid(x)
