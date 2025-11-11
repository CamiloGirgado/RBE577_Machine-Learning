import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import os
#from tqdm import tqdm

# Dataset

class DubinsDataset(Dataset):
    def __init__(self, num_samples=2500, seq_len=50):
        self.data = []
        self.seq_len = seq_len
        for _ in range(num_samples):
            traj, cond = self._generate_sample(seq_len)
            self.data.append({"traj": traj, "cond": cond})

    def _generate_sample(self, seq_len):
        # Random start and end
        start_pos = np.random.uniform(-5, 5, size=(2,))
        start_yaw = np.random.uniform(-np.pi, np.pi)
        end_pos = np.random.uniform(-5, 5, size=(2,))
        curvature = np.random.uniform(0.05, 0.3)

        t = np.linspace(0, 1, seq_len)
        x = np.linspace(start_pos[0], end_pos[0], seq_len)
        y = np.linspace(start_pos[1], end_pos[1], seq_len)
        yaw = start_yaw + curvature * np.sin(2 * np.pi * t)
        traj = np.stack([x, y, yaw], axis=-1)
        cond = np.array([start_pos[0], start_pos[1], end_pos[0], end_pos[1]], dtype=np.float32)
        return traj.astype(np.float32), cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch):
    trajs = np.stack([b['traj'] for b in batch], axis=0)  # (B, L, 3)
    conds = np.stack([b['cond'] for b in batch], axis=0)  # (B, 4)
    return torch.from_numpy(trajs).float(), torch.from_numpy(conds).float()

# Model

class DubinsLSTM(nn.Module):
    def __init__(self, input_dim=3, cond_dim=4, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_cond = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, conds, target_seq=None, teacher_forcing_ratio=0.5):
        B = conds.size(0)
        device = conds.device
        cond_embed = self.fc_cond(conds)
        cond_embed = cond_embed.unsqueeze(1)  # (B, 1, hidden_dim)
        seq_len = target_seq.size(1) if target_seq is not None else 50
        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        out_seq = []

        # initialize input with zeros
        prev_out = torch.zeros(B, 1, 3, device=device)
        for t in range(seq_len):
            rnn_in = torch.cat([prev_out, cond_embed], dim=-1)
            out, (h, c) = self.lstm(rnn_in, (h, c))
            pred = self.fc_out(out)
            out_seq.append(pred)

            if target_seq is not None and random.random() < teacher_forcing_ratio:
                prev_out = target_seq[:, t:t+1, :]
            else:
                prev_out = pred

        return torch.cat(out_seq, dim=1)

# Training and Evaluation

def ADE(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def FDE(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1))


def train_epoch(model, loader, optim, device, criterion, teacher_forcing=0.5, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    for trajs, conds in loader:
        trajs = trajs.float().to(device)
        conds = conds.float().to(device)

        optim.zero_grad()
        preds = model(conds, target_seq=trajs, teacher_forcing_ratio=teacher_forcing)
        loss = criterion(preds, trajs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()
        running_loss += float(loss.item()) * trajs.size(0)
    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    total_ADE = 0.0
    total_FDE = 0.0
    with torch.no_grad():
        for trajs, conds in loader:
            trajs = trajs.float().to(device)
            conds = conds.float().to(device)

            preds = model(conds, target_seq=None, teacher_forcing_ratio=0.0)
            loss = criterion(preds, trajs)
            running_loss += float(loss.item()) * trajs.size(0)
            total_ADE += float(ADE(preds, trajs)) * trajs.size(0)
            total_FDE += float(FDE(preds, trajs)) * trajs.size(0)
    n = len(loader.dataset)
    return running_loss / n, total_ADE / n, total_FDE / n


def plot_prediction_example(model, dataset, device='cpu', idx=0, seq_len=50):
    model.eval()
    trajs, conds = collate_batch([dataset[idx]])
    trajs = trajs.float().to(device)
    conds = conds.float().to(device)
    with torch.no_grad():
        preds = model(conds, target_seq=None, teacher_forcing_ratio=0.0)
    preds = preds.cpu().numpy()[0]
    gt = trajs.cpu().numpy()[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:,0], gt[:,1], gt[:,2], 'b.-', label='gt')
    ax.plot(preds[:,0], preds[:,1], preds[:,2], 'r.-', label='pred')
    ax.scatter([0],[0],[0], c='k', marker='*', s=80, label='start')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Main Training Loop

def main_train(num_samples=2500, seq_len=50, batch_size=64, epochs=30, lr=1e-3, tf_ratio=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating {num_samples} samples...")
    dataset = DubinsDataset(num_samples=num_samples, seq_len=seq_len)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = DubinsLSTM().to(device)
    optim_obj = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'ADE': [], 'FDE': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optim_obj, device, criterion,
                                 teacher_forcing=tf_ratio, clip_grad=2.0)
        val_loss, val_ADE, val_FDE = eval_epoch(model, val_loader, device, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ADE'].append(val_ADE)
        history['FDE'].append(val_FDE)

        print(f"[{epoch+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| ADE: {val_ADE:.3f} | FDE: {val_FDE:.3f}")

    torch.save(model.state_dict(), "dubin_lstm.pth")
    print("Training complete. Model saved as 'dubin_lstm.pth'.")

    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    return model, dataset, train_loader, val_loader, history


if __name__ == "__main__":
    model, dataset, train_loader, val_loader, history = main_train()
    plot_prediction_example(model, dataset)
