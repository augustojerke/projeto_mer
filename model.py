import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    """Bloco residual 2-camadas. Funciona com qualquer tamanho de entrada."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x), inplace=True)


class MERCnn(nn.Module):
    """
    CNN residual para espectrogramas de áudio (mel, stft, mfcc ou tri).

    Funciona com qualquer altura de frequência graças ao AdaptiveAvgPool final.
    Saída com Sigmoid pois os labels do PMEmo estão em [0, 1].
    """
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem: extrai features locais sem reduzir muito uma entrada pequena (MFCC=13 bins)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # freq: F/2, time: T/2
        )

        # Blocos residuais com pooling progressivo
        self.block1 = ConvBlock(32, 64)
        self.pool1  = nn.MaxPool2d(2)   # F/4,  T/4

        self.block2 = ConvBlock(64, 128)
        self.pool2  = nn.MaxPool2d(2)   # F/8,  T/8

        self.block3 = ConvBlock(128, 128)
        self.gap    = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.gap(self.block3(x))
        return self.head(x)


class ResNet18MER(nn.Module):
    """
    ResNet18 pré-treinada no ImageNet com fine-tuning para regressão A/V.

    Estratégia:
      - Congela as camadas iniciais (stem + layer1) para preservar detectores
        de baixo nível que transferem bem para espectrogramas.
      - Fine-tuna layer2, layer3, layer4 e a nova cabeça.
      - Adapta o primeiro conv para in_channels arbitrário (1 ou 3).
      - Substitui o classificador por regressão com Sigmoid ([0,1]).
    """
    def __init__(self, in_channels=1):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapta o primeiro conv para in_channels != 3
        if in_channels != 3:
            old_conv   = base.conv1
            new_conv   = nn.Conv2d(in_channels, old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride,
                                   padding=old_conv.padding, bias=False)
            # Inicializa com média dos pesos dos 3 canais ImageNet
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True)
                                      .expand_as(new_conv.weight))
            base.conv1 = new_conv

        # Congela stem e layer1 (features genéricas de baixo nível)
        for p in base.conv1.parameters():   p.requires_grad = False
        for p in base.bn1.parameters():     p.requires_grad = False
        for p in base.layer1.parameters():  p.requires_grad = False

        # Remove o classificador original
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def _in_ch(modo: str) -> int:
    return 3 if modo == "tri" else 1

def build_model(arch: str, modo: str = "mel") -> nn.Module:
    ch = _in_ch(modo)
    if arch == "cnn":      return MERCnn(in_channels=ch)
    if arch == "cnn3spec": return MERCnn(in_channels=3)
    if arch == "resnet18": return ResNet18MER(in_channels=ch)
    raise ValueError(f"Arquitetura desconhecida: {arch}")
