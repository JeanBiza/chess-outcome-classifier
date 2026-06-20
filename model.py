import torch
import torch.nn as nn

class ChessOutcomeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(8192 + 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 3)
        )

    def forward(self, board_planes, extra_features):
        x = self.conv_block(board_planes)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra_features], dim=1)
        return self.fc_block(x)