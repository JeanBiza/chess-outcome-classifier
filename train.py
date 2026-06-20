import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import ChessOutcomeCNN


DATA_PATH = "chess_data.npz"
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 7
MODEL_SAVE_PATH = "best_chess_model.pt"

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)
    boards = data['boards']
    extra  = data['extra']
    labels = data['labels'] 

    indices = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )

    X_boards_train = torch.tensor(boards[idx_train], dtype=torch.float32)
    X_extra_train  = torch.tensor(extra[idx_train],  dtype=torch.float32)
    y_train        = torch.tensor(labels[idx_train], dtype=torch.long)

    X_boards_test  = torch.tensor(boards[idx_test],  dtype=torch.float32)
    X_extra_test   = torch.tensor(extra[idx_test],   dtype=torch.float32)
    y_test         = torch.tensor(labels[idx_test],  dtype=torch.long)

    print(f"Training samples: {len(y_train)} | Testing samples: {len(y_test)}")

    train_loader = DataLoader(
        TensorDataset(X_boards_train, X_extra_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_boards_test, X_extra_test, y_test),
        batch_size=BATCH_SIZE
    )

    model = ChessOutcomeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3
    )

    best_test_loss = float('inf')
    epochs_no_improve = 0
    print("\n--- Starting Training & Evaluation ---")

    for epoch in range(EPOCHS):

        model.train()
        running_train_loss = 0.0

        for board_batch, extra_batch, label_batch in train_loader:
            board_batch  = board_batch.to(device)
            extra_batch  = extra_batch.to(device)
            label_batch  = label_batch.to(device)
            optimizer.zero_grad()
            predictions = model(board_batch, extra_batch)
            loss = criterion(predictions, label_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)


        model.eval()
        correct_predictions = 0
        total_predictions = 0
        running_test_loss = 0.0

        with torch.no_grad():
            for board_batch, extra_batch, label_batch in test_loader:
                board_batch = board_batch.to(device)
                extra_batch = extra_batch.to(device)
                label_batch = label_batch.to(device)

                predictions = model(board_batch, extra_batch)
                running_test_loss += criterion(predictions, label_batch).item()

                _, predicted_class = torch.max(predictions, 1)
                total_predictions += label_batch.size(0)
                correct_predictions += (predicted_class == label_batch).sum().item()

        avg_test_loss = running_test_loss / len(test_loader)
        accuracy = (correct_predictions / total_predictions) * 100
        current_lr    = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Acc: {accuracy:.2f} | "
              f"LR: {current_lr:.6f}")
        
        scheduler.step(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   -> Best model saved (loss: {best_test_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"   -> No improvement. Patience: {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Early stopping triggered, Training halted to prevent overfitting.")
                break
        
    
    print(f"\nFinised. The best model was saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()