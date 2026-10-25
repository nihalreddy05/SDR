"""Dual-Task 1D CNN: Head 1 modulation classification, Head 2 threat audit."""
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + r)

class DualTaskClassifier(nn.Module):
    """Input: [B, 2, L]. Returns (mod_logits[B, 4], threat_logits[B, 2])."""
    def __init__(self, num_mod_classes: int = 4, input_channels: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res1 = ResidualBlock1D(32)
        self.down1 = nn.Sequential(nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.res2 = ResidualBlock1D(64)
        self.down2 = nn.Sequential(nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.res3 = ResidualBlock1D(128)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.mod_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_mod_classes))
        self.threat_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        out = self.stem(x)
        out = self.res1(out)
        out = self.down1(out)
        out = self.res2(out)
        out = self.down2(out)
        out = self.res3(out)
        out = self.gap(out).squeeze(-1)
        return self.mod_head(out), self.threat_head(out)