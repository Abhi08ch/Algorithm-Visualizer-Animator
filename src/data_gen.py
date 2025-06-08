import numpy as np
from src.algorithms import bubble_sort_steps
import os

def generate_bubble_sort_data(num_samples=1000, arr_len=8, save_path="data/bubble_sort.npy"):
    X, y = [], []
    for _ in range(num_samples):
        arr = np.random.permutation(arr_len)
        steps = list(bubble_sort_steps(arr.tolist()))
        for i in range(len(steps)-1):
            X.append(steps[i])
            y.append(steps[i+1])
    X, y = np.array(X), np.array(y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"X": X, "y": y})
    print(f"Saved {len(X)} samples to {save_path}")

if __name__ == "__main__":
    generate_bubble_sort_data()