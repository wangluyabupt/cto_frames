import torch
from torch import nn
from torch.nn import functional as F

from . import extractors
import pdb


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule,self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=10, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super(PSPNet,self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

##        self.classifier = nn.Sequential(
##            nn.Linear(deep_features_size, 256),
##            nn.ReLU(),
##            nn.Linear(in_features=256, out_features=n_classes)
##        )
        
        self.expand = nn.Conv2d(512, 1024, 1, stride=1)

    def forward(self, x):
        f, x0,x1,x2,x3= self.feats(x)
        #class_f=x3 
        p = self.psp(f)
        p = self.drop_1(p)
        
        x2 = self.expand(x2)
        p0 = p
        p = p + x2
     
        p = self.up_1(p)
        p = self.drop_2(p)
        p1 = p
        p = p + x1

        p = self.up_2(p)
        p = self.drop_2(p)
        p2 = p
        p = p + x0

        p = self.up_3(p)
        p = self.drop_2(p)
        p3 = p
        #pdb.set_trace()

        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p)#, self.classifier(auxiliary)
