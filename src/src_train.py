import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src_ml_model import BubbleSortLSTM

def train_bubble_sort_model(
    data_path="data/bubble_sort.npy",
    model_path="models/bubble_sort_lstm.pt",
    hidden_size=64,
    epochs=15,
    batch_size=64
):
    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    X, y = data["X"], data["y"]
    X = X / (X.shape[1] - 1)
    y = y / (y.shape[1] - 1)
    X = X[..., np.newaxis]  # (samples, seq_len, 1)
    y = y  # (samples, seq_len)

    # Split into train/val
    idx = int(0.9 * len(X))
    X_train, X_val = X[:idx], X[idx:]
    y_train, y_val = y[:idx], y[idx:]

    # Create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    input_size = X.shape[1]
    model = BubbleSortLSTM(input_size=input_size, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_dl.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(val_dl.dataset)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_bubble_sort_model()