import torch
import torch.nn as nn
import numpy as np

class BubbleSortLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BubbleSortLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output
        out = self.fc(out)
        return out

def load_model(model_path, input_size, hidden_size):
    model = BubbleSortLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_next_step(model, arr):
    arr = np.array(arr) / (len(arr) - 1)
    arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # shape: (1, seq_len, 1)
    with torch.no_grad():
        pred = model(arr).squeeze(0).numpy()
    pred = np.argsort(pred)
    return pred.tolist()