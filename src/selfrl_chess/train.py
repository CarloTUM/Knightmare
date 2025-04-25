#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm

# Module-Skripts importieren
from config import DATA_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
from network import PolicyValueNet


class ChessDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length = f["states"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            state = f["states"][idx]
            policy = f["policies"][idx]
            value = f["values"][idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(policy).float(),
            torch.tensor(value).float(),
        )


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Alle Replay-Dateien
    files = sorted(os.listdir(DATA_DIR))
    replay_files = [os.path.join(DATA_DIR, f) for f in files if f.startswith("replay_") and f.endswith(".h5")]
    if not replay_files:
        print("No replay data found in", DATA_DIR)
        return

    model = PolicyValueNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    ce_loss = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, cnt = 0.0, 0
        for h5_path in replay_files:
            dataset = ChessDataset(h5_path)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for states, policies, values in tqdm(loader, desc=f"Epoch {epoch}"):
                states, policies, values = states.to(DEVICE), policies.to(DEVICE), values.to(DEVICE)
                logp, pred_v = model(states)
                pred_v = pred_v.view(-1)

                loss_v = mse_loss(pred_v, values)
                loss_p = ce_loss(logp, policies)
                loss = loss_v + loss_p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * states.size(0)
                cnt += states.size(0)

        avg = total_loss / cnt
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg:.4f}")
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"checkpoint_epoch{epoch}.pth"))

    # Bestes Modell speichern
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
    print("Training complete. Best model saved.")


if __name__ == "__main__":
    train()
