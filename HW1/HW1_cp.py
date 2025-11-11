# save as train_nn_allocator.py and run with: python train_nn_allocator.py
import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# ---------------------------
# Config / hyperparameters
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_SAMPLES = int(200_000)   # change to 1_000_000 for full paper dataset; smaller for quick runs
SEQ_LEN = 16                 # mini-batch sequence length (paper uses sequences per mini-batch)
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization
EPOCHS = 100
PATIENCE = 8                 # early stopping patience
GRAD_CLIP = 1.0

# Loss scaling factors — paper's Table 2 suggests specific scalings.
lambda_alloc = 1.0        # allocation (force matching) loss
lambda_mag   = 1e0        # magnitude penalty weight
lambda_rate  = 1e-7       # rate penalty weight (paper used small numbers)
lambda_power = 1e-7       # power penalty weight
lambda_sector = 1e-1      # sector (forbidden-angle) penalty
# You may replace the above with paper exact table values if you want. :contentReference[oaicite:4]{index=4}

# ---------------------------
# Thruster ranges and mapping
# ---------------------------
# From paper eq (7) example ranges (the paper uses these ranges for the training set):
# tau1 ∈ [-10000, 10000] N
# tau2 ∈ [-5000, 5000] N ; psi2 ∈ [-180, 180] deg
# tau3 ∈ [-5000, 5000] N ; psi3 ∈ [-180, 180] deg
TAU1_RANGE = (-10000.0, 10000.0)
TAU2_RANGE = (-5000.0, 5000.0)
TAU3_RANGE = (-5000.0, 5000.0)
PSI2_RANGE = (-180.0, 180.0)
PSI3_RANGE = (-180.0, 180.0)

# Placeholder mapping matrix B: maps thruster commands (tau1, tau2, tau2_angle, tau3, tau3_angle)
# into generalized forces [X, Y, N]. In practice, you compute B from thruster locations/angles.
# The paper calculates generalized forces via the thruster geometry; replace the example below with your B.
# Here we will compute generalized forces numerically from tau and angle using geometry (more direct).
# Example geometry values (placeholder; replace with the real vessel thruster locations and lever arms):
r2 = np.array([ 5.0, 0.0])  # thruster 2 position relative to CG (x,y) [m] (example)
r3 = np.array([-5.0, 0.0])  # thruster 3
# tunnel thruster 1 is assumed along transverse direction near centerline (force in Y only)

# max rate constraints from Table 1 (used in rate penalty)
MAX_THRUSTER_FORCE_RATE = 1000.0   # N/s (table 1)
MAX_AZIMUTH_RATE = 10.0            # deg/s

# ---------------------------
# Data generation utils
# ---------------------------
def random_walk_sequences(num_samples, seq_len):
    """
    Generate sequences of thruster commands using a random-walk per thruster variable.
    Each sequence is shape (seq_len, num_command_vars) where command vars =
    [tau1, tau2, psi2, tau3, psi3]
    """
    num_steps = num_samples
    # start at zeros
    cmds = np.zeros((num_steps, 5), dtype=np.float32)
    # random step sizes tuned to give meaningful coverage within ranges
    step_sigma = np.array([200.0, 100.0, 6.0, 100.0, 6.0])  # tuned for smooth random walk
    # initialize first sample randomly within ranges
    cmds[0,0] = np.random.uniform(*TAU1_RANGE)
    cmds[0,1] = np.random.uniform(*TAU2_RANGE)
    cmds[0,2] = np.random.uniform(*PSI2_RANGE)
    cmds[0,3] = np.random.uniform(*TAU3_RANGE)
    cmds[0,4] = np.random.uniform(*PSI3_RANGE)
    for i in range(1, num_steps):
        step = np.random.randn(5) * step_sigma
        cmds[i] = cmds[i-1] + step
        # clip to allowed ranges
        cmds[i,0] = np.clip(cmds[i,0], *TAU1_RANGE)
        cmds[i,1] = np.clip(cmds[i,1], *TAU2_RANGE)
        cmds[i,2] = np.clip(cmds[i,2], *PSI2_RANGE)
        cmds[i,3] = np.clip(cmds[i,3], *TAU3_RANGE)
        cmds[i,4] = np.clip(cmds[i,4], *PSI3_RANGE)
    # cut into sequences
    num_seqs = num_steps // seq_len
    cmds = cmds[:num_seqs*seq_len].reshape(num_seqs, seq_len, 5)
    return cmds

def thruster_commands_to_generalized_forces(cmds):
    """
    Convert thruster commands to generalized forces: for each time step compute
    [Fx, Fy, Mz] produced by thrusters.
    cmds shape: (..., 5) with columns [tau1, tau2, psi2_deg, tau3, psi3_deg]
    We'll assume:
      - tau1 produces lateral force (tunnel thruster across Y axis)
      - tau2 and tau3 azimuth thrusters produce forces with direction given by psi and moment = r x F
    Replace geometry (r2, r3) with real thruster positions/angles if available.
    """
    tau1 = cmds[...,0]
    tau2 = cmds[...,1]
    psi2 = np.deg2rad(cmds[...,2])
    tau3 = cmds[...,3]
    psi3 = np.deg2rad(cmds[...,4])

    # thruster 1: tunnel thruster pointing in +/- Y direction (force on Y)
    F1x = np.zeros_like(tau1)
    F1y = tau1

    # thruster 2:
    F2x = tau2 * np.cos(psi2)
    F2y = tau2 * np.sin(psi2)
    # torque z from thruster force: tau_z = r_x * F_y - r_y * F_x  (2D r cross F)
    M2 = r2[0]*F2y - r2[1]*F2x

    # thruster 3:
    F3x = tau3 * np.cos(psi3)
    F3y = tau3 * np.sin(psi3)
    M3 = r3[0]*F3y - r3[1]*F3x

    Fx = F1x + F2x + F3x
    Fy = F1y + F2y + F3y
    Mz = M2 + M3
    gen = np.stack([Fx, Fy, Mz], axis=-1)
    return gen.astype(np.float32)

# ---------------------------
# Dataset and DataLoader
# ---------------------------
class ThrusterDataset(Dataset):
    def __init__(self, sequences_cmds, sequences_gen):
        # sequences_cmds: (Nseq, seq_len, 5)
        # sequences_gen: (Nseq, seq_len, 3)
        self.x = torch.from_numpy(sequences_gen)  # inputs to allocator = generalized forces
        self.y = torch.from_numpy(sequences_cmds) # targets = thruster commands (encoder outputs)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------------------
# Model: Autoencoder-like LSTM (encoder->latent->decoder)
# Encoder receives generalized forces (Fx,Fy,Mz) sequence and outputs thruster commands sequence
# We'll implement encoder LSTM followed by fully-connected layers (decoder) applied per time-step (seq->seq)
# ---------------------------
class AllocatorNet(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, latent_dim=64, out_dim=5, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                                    num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # map LSTM hidden to latent per time-step via linear
        self.fc_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # decoder per time-step: latent -> commands
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: [B, T, 3]
        B, T, _ = x.shape
        lstm_out, _ = self.encoder_lstm(x)     # [B, T, hidden_dim]
        # apply fc_latent time-distributed (BatchNorm1d wants (B*T, latent_dim))
        lstm_out_flat = lstm_out.contiguous().view(B*T, -1)
        latent = self.fc_latent(lstm_out_flat) # [B*T, latent_dim]
        # decode
        out_flat = self.decoder(latent)        # [B*T, out_dim]
        out = out_flat.view(B, T, -1)          # [B, T, out_dim]
        return out

# ---------------------------
# Loss functions approximating the paper
# ---------------------------
def allocation_loss(pred_cmds, gen_force_targets):
    """
    pred_cmds: [B, T, 5]  -> thruster commands predicted
    gen_force_targets: [B, T, 3] -> desired generalized forces (input to allocator)
    compute predicted generalized forces from pred_cmds and penalize MSE with targets
    """
    pred_np = pred_cmds.detach().cpu().numpy() if not isinstance(pred_cmds, np.ndarray) else pred_cmds
    # but we need gradients: compute in torch. Implement geometric mapping in torch.
    tau1 = pred_cmds[...,0]
    tau2 = pred_cmds[...,1]
    psi2 = torch.deg2rad(pred_cmds[...,2])
    tau3 = pred_cmds[...,3]
    psi3 = torch.deg2rad(pred_cmds[...,4])
    F1x = torch.zeros_like(tau1)
    F1y = tau1
    F2x = tau2 * torch.cos(psi2)
    F2y = tau2 * torch.sin(psi2)
    M2 = r2[0]*F2y - r2[1]*F2x
    F3x = tau3 * torch.cos(psi3)
    F3y = tau3 * torch.sin(psi3)
    M3 = r3[0]*F3y - r3[1]*F3x
    Fx = F1x + F2x + F3x
    Fy = F1y + F2y + F3y
    Mz = M2 + M3
    pred_gen = torch.stack([Fx, Fy, Mz], dim=-1)
    # MSE
    return nn.functional.mse_loss(pred_gen, gen_force_targets)

def magnitude_penalty(pred_cmds):
    # penalize commands outside allowed magnitudes (soft penalty using relu)
    # thruster magnitudes: tau1, tau2, tau3
    tau1 = pred_cmds[...,0]
    tau2 = pred_cmds[...,1]
    tau3 = pred_cmds[...,3]
    p = torch.mean(torch.relu(torch.abs(tau1) - TAU1_RANGE[1])**2)
    p += torch.mean(torch.relu(torch.abs(tau2) - TAU2_RANGE[1])**2)
    p += torch.mean(torch.relu(torch.abs(tau3) - TAU3_RANGE[1])**2)
    return p

def rate_penalty(pred_cmds):
    # penalize high rate-of-change between consecutive time steps
    # compute diffs along time axis
    diffs = pred_cmds[:,1:,:] - pred_cmds[:,:-1,:]  # [B, T-1, 5]
    # penalty for torque/force rates exceeding MAX_THRUSTER_FORCE_RATE and azimuth rate exceeding MAX_AZIMUTH_RATE
    # check force rates: columns 0,1,3
    force_rate = torch.abs(diffs[..., [0,1,3]])
    rate_pen_force = torch.mean(torch.relu(force_rate - MAX_THRUSTER_FORCE_RATE)**2)
    az_rate = torch.abs(diffs[..., [2,4]])  # deg differences
    rate_pen_az = torch.mean(torch.relu(az_rate - MAX_AZIMUTH_RATE)**2)
    return rate_pen_force + rate_pen_az

def power_penalty(pred_cmds):
    # approximate power penalty: P ~ RPM^3, and thrust ~ RPM^2 => P ~ (|tau|)^(3/2)
    # apply to thruster magnitudes (tau1, tau2, tau3)
    mag = torch.abs(pred_cmds[..., [0,1,3]])  # [B, T, 3]
    # compute element-wise mag^(3/2)
    p = torch.mean(mag**1.5)
    return p

def sector_penalty(pred_cmds, forbidden_intervals=None):
    """
    penalize thruster azimuth angles that fall into forbidden sectors.
    forbidden_intervals: dict with thruster index -> list of (low_deg, high_deg) forbidden windows
    e.g., forbidden_intervals = {2: [(-20, 20)], 3: [(170, 180), (-180, -170)]}
    We'll implement a soft penalty: distance to nearest allowed angle if inside forbidden interval.
    """
    if forbidden_intervals is None:
        # example: disallowed sectors for thruster 2 and 3 (from figures in the paper).
        forbidden_intervals = {2: [(-30,30)], 4: [(-40, -10), (10,40)]}  # columns 2 and 4 in pred_cmds
    total = 0.0
    count = 0
    for col_idx, intervals in forbidden_intervals.items():
        angles = pred_cmds[..., col_idx]  # degrees
        # compute penalty if angle inside any forbidden interval
        pen = torch.zeros_like(angles)
        for (low, high) in intervals:
            # mask where angle between low and high (handle wrap-around)
            # simple approach assuming low<high and ranges within [-180,180] used in training
            mask = (angles >= low) & (angles <= high)
            # penalty is squared distance to nearest boundary
            dist_low = torch.relu(low - angles)
            dist_high = torch.relu(angles - high)
            dist = torch.where(mask, torch.minimum(angles - low, high - angles), torch.zeros_like(angles))
            # use (how deep inside sector)^2
            pen += torch.where(mask, (torch.minimum(angles - low, high - angles))**2, torch.zeros_like(angles))
        total = total + torch.mean(pen)
        count += 1
    return total / max(1, count)

# ---------------------------
# Training utilities
# ---------------------------
def train_epoch(model, dl, optimizer):
    model.train()
    total_loss = 0.0
    for gen_forces, target_cmds in dl:
        gen_forces = gen_forces.to(DEVICE)      # [B, T, 3]
        target_cmds = target_cmds.to(DEVICE)    # [B, T, 5]
        optimizer.zero_grad()
        pred_cmds = model(gen_forces)           # [B, T, 5]
        # Primary allocation loss
        L_alloc = allocation_loss(pred_cmds, gen_forces)
        L_mag = magnitude_penalty(pred_cmds)
        L_rate = rate_penalty(pred_cmds)
        L_power = power_penalty(pred_cmds)
        L_sector = sector_penalty(pred_cmds)
        loss = (lambda_alloc*L_alloc +
                lambda_mag*L_mag +
                lambda_rate*L_rate +
                lambda_power*L_power +
                lambda_sector*L_sector)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * gen_forces.size(0)
    return total_loss / len(dl.dataset)

def eval_epoch(model, dl):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for gen_forces, target_cmds in dl:
            gen_forces = gen_forces.to(DEVICE)
            target_cmds = target_cmds.to(DEVICE)
            pred_cmds = model(gen_forces)
            L_alloc = allocation_loss(pred_cmds, gen_forces)
            L_mag = magnitude_penalty(pred_cmds)
            L_rate = rate_penalty(pred_cmds)
            L_power = power_penalty(pred_cmds)
            L_sector = sector_penalty(pred_cmds)
            loss = (lambda_alloc*L_alloc +
                    lambda_mag*L_mag +
                    lambda_rate*L_rate +
                    lambda_power*L_power +
                    lambda_sector*L_sector)
            total_loss += loss.item() * gen_forces.size(0)
    return total_loss / len(dl.dataset)

# ---------------------------
# Main: generate dataset, create model, train
# ---------------------------
def main():
    print("Generating data (this may take a moment)...")
    seqs_cmds = random_walk_sequences(NUM_SAMPLES, SEQ_LEN)  # (Nseq, seq_len, 5)
    seqs_gen = thruster_commands_to_generalized_forces(seqs_cmds)  # (Nseq, seq_len, 3)
    print(f"Generated {seqs_cmds.shape[0]} sequences of length {SEQ_LEN}")

    # dataset and split 80/20
    dataset = ThrusterDataset(seqs_cmds, seqs_gen)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = AllocatorNet(in_dim=3, hidden_dim=128, latent_dim=64, out_dim=5, num_layers=2, dropout=0.2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, ) if True else 0.0
        val_loss = eval_epoch(model, val_dl)
        scheduler.step(val_loss)
        t1 = time.time()
        print(f"Epoch {epoch:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={t1-t0:.1f}s")
        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_allocator.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break
    print("Training finished in {:.1f}s. Best val loss: {:.6f}".format(time.time()-start_time, best_val))
    print("Saved best model to best_allocator.pt")

if __name__ == "__main__":
    main()
