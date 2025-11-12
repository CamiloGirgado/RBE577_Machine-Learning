#!/usr/bin/env python3
"""
Dubins Airplane Trajectory Prediction with LSTM
------------------------------------------------

This single script:
1. Uses dubinEHF3d() to generate synthetic 3D Dubins airplane trajectories.
2. Trains a Recurrent Neural Network (LSTM) to predict trajectories.
3. Logs train/val losses with TensorBoard.
4. Saves 10 plots comparing ground-truth vs. predicted trajectories.

Run:
    python dubin_LSTM_merged.py

View logs:
    tensorboard --logdir runs

Dependencies:
    numpy
    matplotlib
    torch
    tqdm
    tensorboard
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------
# Dubins airplane generator (from dubinEHF3d.py)
# ---------------------------------------------------
def dubinEHF3d(east1, north1, alt1, psi1, east2, north2, r, step, gamma):
    MAX_NUM_PATH_POINTS = 1000
    path = np.zeros((MAX_NUM_PATH_POINTS, 3))
    r_sq = r**2
    psi1 %= 2*np.pi

    theta_l = psi1 + np.pi/2
    eastc_l = east1 + r*np.cos(theta_l)
    northc_l = north1 + r*np.sin(theta_l)

    theta_r = psi1 - np.pi/2
    eastc_r = east1 + r*np.cos(theta_r)
    northc_r = north1 + r*np.sin(theta_r)

    d2c_l_sq = (east2-eastc_l)**2 + (north2-northc_l)**2
    d2c_r_sq = (east2-eastc_r)**2 + (north2-northc_r)**2
    d2c_l = np.sqrt(d2c_l_sq)
    d2c_r = np.sqrt(d2c_r_sq)

    if d2c_l < r or d2c_r < r:
        return np.zeros((0, 3)), 0, 0

    theta_c_l = np.arctan2(north2-northc_l, east2-eastc_l)
    theta_c_r = np.arctan2(north2-northc_r, east2-eastc_r)

    lt_l = np.sqrt(d2c_l_sq - r_sq)
    lt_r = np.sqrt(d2c_r_sq - r_sq)

    theta_start_l = theta_r
    theta_start_r = theta_l
    theta_d_l = np.arccos(r/d2c_l)
    theta_end_l = theta_c_l - theta_d_l

    theta_d_r = np.arccos(r/d2c_r)
    theta_end_r = theta_c_r + theta_d_r

    arc_l = abs(theta_end_l - theta_start_l)
    arc_r = abs(theta_end_r - theta_start_r)
    arc_length_l = arc_l * r
    arc_length_r = arc_r * r
    total_length_l = arc_length_l + lt_l
    total_length_r = arc_length_r + lt_r

    if total_length_l < total_length_r:
        theta_step = step/r
        num_arc_seg = max(2, int(np.ceil(arc_l/theta_step)))
        angles = np.linspace(theta_start_l, theta_end_l, num_arc_seg)
        alt_end = alt1 + arc_length_l * np.tan(gamma)
        altitude = np.linspace(alt1, alt_end, num_arc_seg)
        arc_traj = np.column_stack([
            eastc_l + r*np.cos(angles),
            northc_l + r*np.sin(angles),
            altitude
        ])
        num_line_seg = max(2, int(np.ceil(lt_l/step)))
        alt_begin = arc_traj[-1, 2]
        alt_end = alt_begin + lt_l * np.tan(gamma)
        line_traj = np.column_stack([
            np.linspace(arc_traj[-1, 0], east2, num_line_seg),
            np.linspace(arc_traj[-1, 1], north2, num_line_seg),
            np.linspace(alt_begin, alt_end, num_line_seg)
        ])
        num_path_points = num_arc_seg + num_line_seg - 1
        path[:num_arc_seg, :] = arc_traj
        path[num_arc_seg:num_path_points, :] = line_traj[1:, :]
    else:
        theta_step = step/r
        num_arc_seg = max(2, int(np.ceil(arc_r/theta_step)))
        angles = np.linspace(theta_start_r, theta_end_r, num_arc_seg)
        alt_end = alt1 + arc_length_r * np.tan(gamma)
        altitude = np.linspace(alt1, alt_end, num_arc_seg)
        arc_traj = np.column_stack([
            eastc_r + r*np.cos(angles),
            northc_r + r*np.sin(angles),
            altitude
        ])
        num_line_seg = max(2, int(np.ceil(lt_r/step)))
        alt_begin = arc_traj[-1, 2]
        alt_end = alt_begin + lt_r * np.tan(gamma)
        line_traj = np.column_stack([
            np.linspace(arc_traj[-1, 0], east2, num_line_seg),
            np.linspace(arc_traj[-1, 1], north2, num_line_seg),
            np.linspace(alt_begin, alt_end, num_line_seg)
        ])
        num_path_points = num_arc_seg + num_line_seg - 1
        path[:num_arc_seg, :] = arc_traj
        path[num_arc_seg:num_path_points, :] = line_traj[1:, :]

    psi_end = np.arctan2(north2 - path[num_path_points-1, 1], east2 - path[num_path_points-1, 0])
    return path[:num_path_points], psi_end, num_path_points

# ---------------------------------------------------
# Dataset generation
# ---------------------------------------------------
def generate_dataset(num_samples=1000, seq_len=50, r=100, step=10):
    data = []
    for _ in tqdm(range(num_samples), desc="Generating Dubins data"):
        psi = random.uniform(0, 2*np.pi)
        gamma = math.radians(random.uniform(-30, 30))
        goal_x = random.uniform(-500, 500)
        goal_y = random.uniform(-500, 500)
        traj, _, n = dubinEHF3d(0, 0, 0, psi, goal_x, goal_y, r, step, gamma)
        if len(traj) < seq_len:
            continue
        # resample to fixed length
        idx = np.linspace(0, len(traj)-1, seq_len).astype(int)
        traj_fixed = traj[idx]
        data.append(traj_fixed)
    data = np.array(data)
    return data

class DubinsDataset(Dataset):
    def __init__(self, trajectories):
        self.x = torch.tensor(trajectories[:, :-1, :], dtype=torch.float32)
        self.y = torch.tensor(trajectories[:, 1:, :], dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------------------------------------------
# LSTM Model
# ---------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# ---------------------------------------------------
# Training
# ---------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu"):
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch+1}/{epochs}  Train: {train_loss:.5f}  Val: {val_loss:.5f}")

    writer.close()

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------
def plot_predictions(model, dataset, device="cpu", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for i in range(10):
        x, y = dataset[i]
        x, y = x.unsqueeze(0).to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        gt = y.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt[:,0], gt[:,1], gt[:,2], 'b-', label='Ground Truth')
        ax.plot(pred[:,0], pred[:,1], pred[:,2], 'r--', label='Prediction')
        ax.legend()
        ax.set_title(f"Example {i}")
        plt.savefig(f"{save_dir}/traj_{i}.png")
        plt.close()

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = generate_dataset(1200, seq_len=50)
    dataset = DubinsDataset(data)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = LSTMModel().to(device)
    train_model(model, train_loader, val_loader, epochs=60, lr=1e-3, device=device)
    plot_predictions(model, dataset, device)
    print("✅ Training complete. Plots saved to /plots and TensorBoard logs in /runs/")
