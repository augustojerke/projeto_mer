import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights



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

        if in_channels != 3:
            old_conv   = base.conv1
            new_conv   = nn.Conv2d(in_channels, old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride,
                                   padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True)
                                      .expand_as(new_conv.weight))
            base.conv1 = new_conv

        for p in base.conv1.parameters():   p.requires_grad = False
        for p in base.bn1.parameters():     p.requires_grad = False
        for p in base.layer1.parameters():  p.requires_grad = False

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
    if arch == "resnet18": return ResNet18MER(in_channels=ch)
    raise ValueError(f"Arquitetura desconhecida: {arch}")
