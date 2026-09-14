from torch import nn


__all__ = ['DeePSiMNorm', 'DeePSiMNorm2', 'DeePSiMConv34',
           'DeePSiMPool5', 'DeePSiMFc', 'DeePSiMFc8']


class DeePSiMNorm(nn.Module):
    _layer1_ios = (96, 128, 2)

    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.tanh = nn.Tanh()

        # layers
        l1_ios = self._layer1_ios
        self.conv6 = nn.Conv2d(l1_ios[0], l1_ios[1], 3, stride=l1_ios[2], padding=2)
        self.conv7 = nn.Conv2d(l1_ios[1], 128, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.tconv4_0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)
        self.conv4_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.tconv2_0 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.tconv1_0 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.conv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.conv3_1(x))
        x = lrelu(self.tconv2_0(x))
        x = lrelu(self.conv2_1(x))
        x = self.tconv1_0(x)
        x = self.tanh(self.conv1_1(x))
        return x * 255


class DeePSiMNorm2(DeePSiMNorm):
    _layer1_ios = (256, 256, 1)


class DeePSiMConv34(nn.Module):
    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.tanh = nn.Tanh()

        # layers
        self.conv6 = nn.Conv2d(384, 384, 3, padding=0)
        self.conv7 = nn.Conv2d(384, 512, 3, padding=0)
        self.conv8 = nn.Conv2d(512, 512, 2, padding=0)
        self.tconv5_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2_0 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False)
        self.tconv1_0 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2_0(x))
        x = lrelu(self.conv2_1(x))
        x = lrelu(self.tconv1_0(x))
        x = self.tanh(self.conv1_1(x))
        return x * 255


class DeePSiMPool5(nn.Module):
    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        # layers
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=0)
        self.tconv5_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.tconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.conv6(x))
        x = lrelu(self.conv7(x))
        x = lrelu(self.conv8(x))
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2(x))
        x = lrelu(self.tconv1(x))
        x = self.tconv0(x)
        return x


class DeePSiMFc(nn.Module):
    _num_inputs = 4096

    def __init__(self):
        super().__init__()
        # reusable activation funcs
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        # layers
        self.fc7 = nn.Linear(self._num_inputs, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.tconv5_0 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.tconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.fc7(x))
        x = lrelu(self.fc6(x))
        x = lrelu(self.fc5(x))
        x = x.view(-1, 256, 4, 4)
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2(x))
        x = lrelu(self.tconv1(x))
        x = self.tconv0(x)
        return x


class DeePSiMFc8(DeePSiMFc):
    _num_inputs = 1000
