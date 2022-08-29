import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from coordchanatt import CoordAtt as ca


class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.S_net = nn.Sequential(
            # 1                                                3 * 224 * 224
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # 64 * 224 * 224
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # 64 * 224 * 224
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 64 * 112 * 112
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 128 * 112 * 112
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 * 112 * 112
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 128 * 56 * 56
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 * 56 * 56
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 * 56 * 56
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 * 56 *56
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 256 * 28 * 28
        )
        self.ca = ca(256, 256)
        self.C_net = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512 * 28 * 28
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512 * 28 * 28
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 512 * 14 * 14
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 512 * 7 * 7
            # nn.AvgPool2d(kernel_size=1, stride=1),
        )

        # self.avg_pool = nn.AvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, input1, input2):
        self.gradients = []
        output1 = self.S_net(input1)
        output2 = self.S_net(input2)
        output = torch.abs(output1 - output2)  # 通道相减，然后取绝对值
        CA = self.ca(output)
        features = self.C_net(CA)
        # CA.register_hook((self.save_gradient))
        features_flatten = features.view(features.size(0), -1)
        result = self.classifier(features_flatten)
        return result, CA


class Generator(nn.Module):
    def __init__(self, img_shape, latent_z_dim):
        super(Generator, self).__init__()
        self.image_shape = img_shape
        self.latent_z_dim = latent_z_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_layer = nn.Sequential(nn.Linear(512, self.latent_z_dim))
        # self.extra_layer = nn.Conv2d(256, self.latent_z_dim, 28)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_z_dim * 2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, features, z):
        features_avg = self.avg_pool(features).view(features.size(0), -1)
        features_max = self.max_pool(features).view(features.size(0), -1)
        new_features = torch.cat((features_avg, features_max), dim=1)
        disturb = self.mlp_layer(new_features)
        # new_z = disturb + z
        new_z = torch.cat((disturb, z), dim=1)
        # features_flatten = self.extra_layer(features).view(features.size(0), -1)
        # new_z = torch.cat((features_flatten, z), dim=1)
        # new_z = features_flatten + z
        img = self.model(new_z)
        img = img.view(z.size(0), *self.image_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.image_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat).view(-1)
        return validity
