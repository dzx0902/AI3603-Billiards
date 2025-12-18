import numpy as np
import json
import os

BALL_ORDER = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
POCKET_ORDER = ['lb','lc','lt','rb','rc','rt']

def encode_state(balls, my_targets, table):
    cue_pos = balls['cue'].state.rvw[0]
    cue_xy = np.array([cue_pos[0], cue_pos[1]], dtype=np.float32)
    my_set = set(my_targets)
    vec = [cue_xy[0], cue_xy[1]]
    for bid in BALL_ORDER:
        b = balls[bid]
        exist_mask = 0.0 if int(b.state.s) == 4 else 1.0
        pos = b.state.rvw[0]
        x = pos[0] if exist_mask == 1.0 else 0.0
        y = pos[1] if exist_mask == 1.0 else 0.0
        my_mask = 1.0 if bid in my_set else 0.0
        vec.extend([x, y, exist_mask, my_mask])
    vec.extend([float(table.w), float(table.l)])
    for pid in POCKET_ORDER:
        c = table.pockets[pid].center
        vec.extend([float(c[0]), float(c[1])])
    return np.asarray(vec, dtype=np.float32)

def default_meta(state_dim):
    pbounds = {'V0': [0.5, 8.0], 'phi': [0.0, 360.0], 'theta': [0.0, 90.0], 'a': [-0.5, 0.5], 'b': [-0.5, 0.5]}
    return {
        'state_dim': int(state_dim),
        'action_dim': 5,
        'pbounds': pbounds,
        'state_mean': [0.0]*state_dim,
        'state_std': [1.0]*state_dim
    }

def load_meta(meta_path, state_dim):
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if 'state_dim' not in meta:
                meta['state_dim'] = state_dim
            return meta
        except Exception:
            pass
    return default_meta(state_dim)

def normalize_state(state_vec, meta):
    mean = np.asarray(meta.get('state_mean', [0.0]*len(state_vec)), dtype=np.float32)
    std = np.asarray(meta.get('state_std', [1.0]*len(state_vec)), dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std)
    return ((state_vec - mean) / std).astype(np.float32)

def action_denorm(a_tanh, meta):
    pb = meta.get('pbounds', {'V0': [0.5, 8.0], 'phi': [0.0, 360.0], 'theta': [0.0, 90.0], 'a': [-0.5, 0.5], 'b': [-0.5, 0.5]})
    keys = ['V0','phi','theta','a','b']
    out = {}
    for i,k in enumerate(keys):
        lo, hi = float(pb[k][0]), float(pb[k][1])
        v = float((a_tanh[i] + 1.0) * 0.5 * (hi - lo) + lo)
        out[k] = v
    out['phi'] = float(out['phi'] % 360.0)
    out['V0'] = float(np.clip(out['V0'], pb['V0'][0], pb['V0'][1]))
    out['theta'] = float(np.clip(out['theta'], pb['theta'][0], pb['theta'][1]))
    out['a'] = float(np.clip(out['a'], pb['a'][0], pb['a'][1]))
    out['b'] = float(np.clip(out['b'], pb['b'][0], pb['b'][1]))
    return out

def action_norm(action, meta):
    pb = meta.get('pbounds', {'V0': [0.5, 8.0], 'phi': [0.0, 360.0], 'theta': [0.0, 90.0], 'a': [-0.5, 0.5], 'b': [-0.5, 0.5]})
    keys = ['V0','phi','theta','a','b']
    arr = []
    for k in keys:
        lo, hi = float(pb[k][0]), float(pb[k][1])
        v = float(action[k])
        v = (v - lo) / (hi - lo) * 2.0 - 1.0
        arr.append(v)
    return np.asarray(arr, dtype=np.float32)
