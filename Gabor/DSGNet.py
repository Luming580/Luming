from space import *
from Gabor.back.resnext101_regular import ResNeXt101

class DSGNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(DSGNet, self).__init__()
        # params

        # backbone
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.conv5 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # h fusion
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down3 = nn.AvgPool2d((2, 2), stride=2)
        self.a_fusion = CBAM(896)
        self.h_fusion_conv = nn.Sequential(nn.Conv2d(896, 896, 3, 1, 1), nn.BatchNorm2d(896), nn.ReLU())

        # l fusion
        self.l_fusion_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.h2l = nn.ConvTranspose2d(896, 1, 8, 4, 2)

        # final fusion
        self.h_up_for_final_fusion = nn.ConvTranspose2d(896, 256, 8, 4, 2)
        self.final_attention = CBAM(320)
        self.final_fusion_conv = nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), nn.ReLU())

        self.final_predict = nn.Conv2d(320, 1, 3, 1, 1)

    def forward(self, x):
        layer0 = self.layer0(x)  # [-1, 64, 128, 128]
        layer1 = self.layer1(layer0)  # [-1, 256, 64, 64]
        layer2 = self.layer2(layer1)  # [-1, 512, 32, 32]
        layer3 = self.layer3(layer2)  # [-1, 1024, 16, 16]
        layer4 = self.layer4(layer3)  # [-1, 2048, 8, 8]

        conv5 = self.conv5(layer4)  # [-1, 512, 8, 8]
        conv4 = self.conv4(layer3)  # [-1, 256, 16, 16]
        conv3 = self.conv3(layer2)  # [-1, 128, 32, 32]
        conv2 = self.conv2(layer1)  # [-1, 64, 64, 64]

        up5 = self.up5(conv5)  # [-1, 512, 16, 16]
        down3 = self.down3(conv3)  # [-1, 128, 16, 16]
        h_fusion = self.h_fusion_conv(self.a_fusion(torch.cat((up5, conv4, down3), 1)))

        h2l = self.h2l(h_fusion)
        l_fusion = torch.sigmoid(h2l) * self.l_fusion_conv(conv2)

        h_up_for_final_fusion = self.h_up_for_final_fusion(h_fusion)
        final_fusion = self.final_attention(torch.cat((h_up_for_final_fusion, l_fusion), 1))
        final_fusion = self.final_fusion_conv(final_fusion)

        final_predict = self.final_predict(final_fusion)

        final_predict = F.interpolate(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(final_predict)


# if __name__ == '__main__':
#     x = torch.randn(1, 1, 256, 256)
#     model = DSGNet()
#     y = model(x)
#     print(y.shape)
