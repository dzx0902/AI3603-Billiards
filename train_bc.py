import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
from typing import List

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/bc_data.npz')
    parser.add_argument('--out', type=str, default='models/bc_policy.pt')
    parser.add_argument('--meta', type=str, default='models/meta.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--log_dir', type=str, default='models/logs')
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=int, default=1)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--dim_weights', type=str, default='1,1,1,1,1')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    data = np.load(args.data)
    states = data['states'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-6
    states_n = ((states - mean) / std).astype(np.float32)
    in_dim = states_n.shape[1]
    out_dim = actions.shape[1]
    n = states_n.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_n = int(max(1, n * args.val_split))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    states_train = states_n[train_idx]
    actions_train = actions[train_idx]
    states_val = states_n[val_idx]
    actions_val = actions[val_idx]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(in_dim, out_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-5)
    else:
        sched = None
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(states_train), torch.from_numpy(actions_train))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    val_states_t = torch.from_numpy(states_val).to(device)
    val_actions_t = torch.from_numpy(actions_val).to(device)
    train_losses = []
    val_losses = []
    dim_w = torch.tensor([float(x) for x in args.dim_weights.split(',')], dtype=torch.float32, device=device)
    dim_w = dim_w.view(1, -1)
    model.train()
    best_val = float('inf')
    best_state = None
    patience_cnt = 0
    for ep in range(args.epochs):
        running = 0.0
        batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mse = (pred - yb) ** 2
            loss = (mse * dim_w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            batches += 1
        train_loss = running / max(1, batches)
        with torch.no_grad():
            model.eval()
            val_pred = model(val_states_t)
            val_mse = (val_pred - val_actions_t) ** 2
            val_loss = (val_mse * dim_w).mean().item()
            model.train()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[BC] epoch {ep+1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if sched:
            sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"[BC] Early stop at epoch {ep+1}, best_val={best_val:.6f}")
                break
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    scripted = torch.jit.script(model)
    scripted.save(args.out)
    meta = {
        'state_dim': int(in_dim),
        'action_dim': int(out_dim),
        'state_mean': mean.tolist(),
        'state_std': std.tolist(),
        'pbounds': {'V0': [0.5, 8.0], 'phi': [0.0, 360.0], 'theta': [0.0, 90.0], 'a': [-0.5, 0.5], 'b': [-0.5, 0.5]},
    }
    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    metrics = {'train_loss': train_losses, 'val_loss': val_losses}
    with open(os.path.join(args.log_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if _HAS_MPL:
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.title('BC Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.log_dir, 'loss_curve.png'))

if __name__ == '__main__':
    main()
