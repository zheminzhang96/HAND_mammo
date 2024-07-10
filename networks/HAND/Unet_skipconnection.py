import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout2d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) ## to be check

    def forward(self, x):
        y = self.conv(x)

        return y



class Unet(nn.Module):
    def __init__(self, in_channels=1, base_channels=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 4, 256, 256)

        x1_1 = self.EnBlock1(x)     # (1, 4, 256, 256)  --out
        x1_2 = self.EnDown1(x1_1)  # (1, 8, 128, 128)

        x2_1 = self.EnBlock2_1(x1_2)    # (1, 8, 128, 128)  --out
        x2_1 = self.EnBlock2_2(x2_1)    # (1, 8, 128, 128)
        x2_2 = self.EnDown2(x2_1)  # (1, 16, 64, 64)

        x3_1 = self.EnBlock3_1(x2_2)    # (1, 16, 64, 64)  --out
        x3_1 = self.EnBlock3_2(x3_1)    # (1, 16, 64, 64)
        x3_2 = self.EnDown3(x3_1)  # (1, 32, 32, 32)

        x4_1 = self.EnBlock4_1(x3_2)    # (1, 32, 32, 32)
        x4_2 = self.EnBlock4_2(x4_1)    # (1, 32, 32, 32)
        x4_3 = self.EnBlock4_3(x4_2)    # (1, 32, 32, 32)
        output = self.EnBlock4_4(x4_3)  # (1, 32, 32, 32)  --out

        return x1_1,x2_1,x3_1,output


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        devide_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.rand((1, 1, 256, 256), device=devide_id)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=1, base_channels=4)
        model.to(devide_id)
        x11, x21, x31, output = model(x)
        print('x1_1:', x11.shape)   # torch.Size([1, 4, 256, 256])
        print('x2_1:', x21.shape)   # torch.Size([1, 8, 128, 128])
        print('x3_1:', x31.shape)   # torch.Size([1, 16, 64, 64])
        print('output:', output.shape)  # torch.Size([1, 32, 32, 32])
