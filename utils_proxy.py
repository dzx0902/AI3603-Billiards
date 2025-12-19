import numpy as np

def _angle_between(vec, deg):
    v = np.asarray(vec, dtype=np.float32)
    if np.linalg.norm(v) < 1e-8:
        return np.pi
    v = v / (np.linalg.norm(v) + 1e-8)
    ang = np.deg2rad(float(deg))
    d = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-8)
    cos = float(np.clip(np.dot(v, d), -1.0, 1.0))
    return float(np.arccos(cos))

def _nearest_pocket_to(xy, pockets_xy):
    xy = np.asarray(xy, dtype=np.float32)
    d = np.linalg.norm(pockets_xy - xy[None, :], axis=1)
    i = int(np.argmin(d))
    return pockets_xy[i]

def proxy_score(state_vec, action, my_targets_ids):
    s = np.asarray(state_vec, dtype=np.float32)
    cue_xy = s[0:2]
    # balls block starts at index 2, each ball 4 values: x,y,exist,ismine
    balls_blk = s[2:2+15*4].reshape(15, 4)
    # pockets: after table size (2), then 6 pockets xy
    table_blk = s[2+15*4:2+15*4+2]
    pockets_blk = s[2+15*4+2:].reshape(6, 2)
    # select target: nearest existing my ball
    my_mask = balls_blk[:, 3] > 0.5
    exist_mask = balls_blk[:, 2] > 0.5
    sel = np.where(my_mask & exist_mask)[0]
    if sel.size == 0:
        # when only 8 is target, treat black ball index=7
        sel = np.array([7], dtype=np.int64)
    xs = balls_blk[sel, 0:2]
    dists = np.linalg.norm(xs - cue_xy[None, :], axis=1)
    j = int(np.argmin(dists))
    target_xy = xs[j]
    # angle alignment between cue->target and shot phi
    ang_diff = _angle_between(target_xy - cue_xy, action['phi'])
    align_score = float(np.cos(ang_diff))  # [-1,1]
    # pocket alignment: choose nearest pocket to target
    pk = _nearest_pocket_to(target_xy, pockets_blk)
    tgt_dir = pk - target_xy
    # approximate required carom: we still use shot phi for direction proxy
    ang_diff2 = _angle_between(tgt_dir, action['phi'])
    pocket_align = float(np.cos(ang_diff2))
    # distance penalty: shorter cue->target better
    dist_pen = float(dists[j])
    # theta penalty: extreme elevation risky
    theta_pen = float(abs(action['theta'])) / 90.0
    # a/b spin offsets penalty magnitude
    spin_pen = float(np.hypot(action['a'], action['b'])) / 0.5
    # score aggregation (deterministic, cheap)
    score = 1.5 * align_score + 1.2 * pocket_align - 0.3 * dist_pen - 0.4 * theta_pen - 0.2 * spin_pen
    return float(score)
