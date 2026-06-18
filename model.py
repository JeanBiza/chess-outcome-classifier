import torch
import torch.nn as nn

class ChessOutcomeClassifier(nn.Module):
    def __init__(self):
        super(ChessOutcomeClassifier, self).__init__()

        self.fc1 = nn.Linear(in_features=768, out_features=512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(in_features=256, out_features=3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        
        return x