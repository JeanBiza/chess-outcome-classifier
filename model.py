import torch
import torch.nn as nn

class ChessOutcomeClassifier(nn.Module):
    def __init__(self):
        super(ChessOutcomeClassifier, self).__init__()
        
        self.fc1 = nn.Linear(in_features=769, out_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024) # NEW: Batch Normalization
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3) 
        
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)
        
        self.fc4 = nn.Linear(in_features=256, out_features=3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x) 
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x) 
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        return x