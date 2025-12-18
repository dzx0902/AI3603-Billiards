import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)
    data = np.load(args.data)
    states = data['states'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-6
    states_n = ((states - mean) / std).astype(np.float32)
    in_dim = states_n.shape[1]
    out_dim = actions.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(in_dim, out_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(states_n), torch.from_numpy(actions))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    model.train()
    for ep in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
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

if __name__ == '__main__':
    main()
