import torch
from torch import nn
from torch.nn import functional as F
from .vie import VIEOperator


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Tanh(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Tanh(),
        )


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.encoder1 = ConvBlock(input_channels, 128)
        self.encoder2 = ConvBlock(128, 256)
        self.encoder3 = ConvBlock(256, 512)
        self.encoder4 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(1024, 1024)
        self.decoder4 = ConvBlock(2048, 512)
        self.decoder3 = ConvBlock(1024, 256)
        self.decoder2 = ConvBlock(512, 128)
        self.decoder1 = ConvBlock(256, 128)
        self.projection = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.Tanh())
        self.output = nn.Sequential(nn.Conv2d(64, output_channels, 3, padding=1), nn.Tanh())

    def _merge(self, decoder, encoder):
        upsampled = F.interpolate(decoder, size=encoder.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat((encoder, upsampled), dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))
        z = self.decoder4(self._merge(z, e4))
        z = self.decoder3(self._merge(z, e3))
        z = self.decoder2(self._merge(z, e2))
        z = self.decoder1(self._merge(z, e1))
        return self.output(self.projection(z))


def field_to_initial_gain(field):
    amplitude = torch.sqrt(field[:, :1].square() + field[:, 1:2].square() + 1e-9)
    path_loss_db = -20.0 * torch.log10(amplitude + 1e-9)
    bounded = path_loss_db.clamp(-11.0, 70.0)
    return 1.0 - (bounded + 11.0) / 81.0


class PEFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.knowledge = UNet(4, 2)
        self.data = UNet(1, 1)
        self.vie = VIEOperator()

    def forward(self, inputs, incident=None, target=None):
        field = self.knowledge(inputs)
        initial = field_to_initial_gain(field)
        correction = self.data(initial)
        prediction = initial + correction
        if target is None:
            return prediction
        if incident is None:
            raise ValueError('Incident electric field is required for training')
        residual = self.vie.residual(field.float(), incident.float(), inputs[:, 1:2])
        pixels = target.shape[-2] * target.shape[-1]
        physical_loss = self.vie.physical_loss(residual)
        data_loss = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1).div(pixels).mean()
        return prediction, physical_loss, data_loss, physical_loss + data_loss
