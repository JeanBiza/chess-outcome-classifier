import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import ChessOutcomeClassifier

DATA_PATH = "chess_data.npz"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
PATIENCE = 3
MODEL_SAVE_PATH = "best_chess_model.pt"

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        data['features'], data['labels'], test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

    model = ChessOutcomeClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    best_test_loss = float('inf')
    epochs_no_improve = 0

    print("\n--- Starting Training & Evaluation ---")

    for epoch in range(EPOCHS):

        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)


        model.eval()
        correct_predictions = 0
        total_predictions = 0
        running_test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                predictions = model(inputs)
                loss = criterion(predictions, labels)
                running_test_loss += loss.item()

                _, predicted_class = torch.max(predictions, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted_class == labels).sum().item()

        avg_test_loss = running_test_loss / len(test_loader)
        accuracy = (correct_predictions / total_predictions) * 100

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Test Loss improved, New best model saved.")
        else:
            epochs_no_improve += 1
            print(f" -> No improvement. Patience: {epochs_no_improve}/{PATIENCE}")

            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Early stopping triggered, Training halted to prevent overfitting.")
                break
        
    
    print(f"\nFinised. The best model was saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()