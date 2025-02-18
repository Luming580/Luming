from Gabor.GFwithSobel import GaborFilter
from Gabor.DSGNet import DSGNet
from Gabor.space import *

class Generator_com_diff(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(Generator_com_diff, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down11 = Gabor_block(ch_in=img_ch, ch_out=64, t=t)
        self.down21 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.down31 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.down41 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.down51 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.gf1 = GaborFilter(dim=1024, size=16, num=2, len=32, number=4)
        self.gf2 = GaborFilter(dim=512, size=32, num=2, len=64, number=8)
        self.gf3 = GaborFilter(dim=256, size=64, num=2, len=128, number=16)
        self.gf4 = GaborFilter(dim=128, size=128, num=2, len=256, number=32)

        self.Up51 = up_conv(ch_in=1024, ch_out=512)
        self.updown51 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up41 = up_conv(ch_in=512, ch_out=256)
        self.updown61 = RRCNN_block(ch_in=256 * 2, ch_out=256, t=t)

        self.Up31 = up_conv(ch_in=256, ch_out=128)
        self.updown71 = RRCNN_block(ch_in=128 * 2, ch_out=128, t=t)

        self.Up21 = up_conv(ch_in=128, ch_out=64)
        self.updown81 = RRCNN_block(ch_in=64 * 2, ch_out=64, t=t)

        self.Conv_1x1_1 = GaborConv2d(64, output_ch, kernel_size=(1, 1), stride=1, padding=0)

        # --------------------------------------------------------------------------------------------------------
        self.segmentation = DSGNet()

    def forward(self, x):
        x1 = self.down11(x)  # torch.Size([1, 64, 256, 256])
        x2 = self.down21(self.Maxpool(x1))  # torch.Size([1, 128, 128, 128])
        x3 = self.down31(self.Maxpool(x2))  # torch.Size([1, 256, 64, 64])
        x4 = self.down41(self.Maxpool(x3))  # torch.Size([1, 512, 32, 32])
        x5 = self.down51(self.Maxpool(x4))  # torch.Size([1, 1024, 16, 16])

        x6 = self.Up51(x5)  # torch.Size([1, 512, 32, 32])
        x6 = torch.cat([x4, x6], dim=1)  # torch.Size([1, 512*2, 32, 32])
        x6 = self.gf1(x6)
        x6 = self.updown51(x6)  # torch.Size([1, 512, 32, 32])

        x7 = self.Up41(x6)  # torch.Size([1, 256, 64, 64])
        x7 = torch.cat([x7, x3], dim=1)  # torch.Size([1, 256*2, 64, 64])
        x7 = self.gf2(x7)
        x7 = self.updown61(x7)  # torch.Size([1, 256, 64, 64])

        x8 = self.Up31(x7)  # torch.Size([1, 128, 128, 128])
        x8 = torch.cat([x8, x2], dim=1)  # torch.Size([1, 128*2, 128, 128])
        x8 = self.gf3(x8)
        x8 = self.updown71(x8)  # torch.Size([1, 128, 128, 128])

        x9 = self.Up21(x8)  # torch.Size([1, 64, 256, 256])
        x9 = torch.cat([x9, x1], dim=1)  # torch.Size([1, 64*2, 256, 256])
        x9 = self.gf4(x9)
        x9 = self.updown81(x9)  # torch.Size([1, 64, 256, 256])

        mid = self.Conv_1x1_1(x9)  # torch.Size([1, 1, 256, 256])
        # -------------------------------------------------------------------------------------------------------------

        fake_target = self.segmentation(mid)

        return fake_target, mid


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generate_CommonAndDiff = Generator_com_diff()

    def forward(self, x):
        pred_gt, Common_Picture = self.generate_CommonAndDiff(x)
        return pred_gt, Common_Picture


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

if __name__ == '__main__':
    net = Generator().cuda()
    x = torch.randn(1, 1, 256, 256)
    a, b = net(x.cuda())
    print(a.shape, b.shape)
