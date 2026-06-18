import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from model import ChessOutcomeClassifier

DATA_PATH = "chess_data.npz"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1

def run_mini_test():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"Using device: {device}")

    print("Loading data from disk...")
    data = np.load(DATA_PATH)
    X_np = data['features']
    y_np = data['labels']

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Total batches to process: {len(dataloader)}")

    model = ChessOutcomeClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() 

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if batch_index % 100 == 99:
                avg_loss = running_loss / 100
                print(f"Batch [{batch_index + 1}/{len(dataloader)}] - Loss: {avg_loss:.4f}")
                running_loss = 0.0
            
    print(f"\nTest complete. The model successfully learned from the data.")

if __name__ == "__main__":
    run_mini_test()