import numpy as np
import copy
import multiprocessing as mp
import pooltool as pt
from collections import OrderedDict
from utils_state import encode_state
from utils_proxy import proxy_score
from agent import analyze_shot_for_reward

class LRUCache:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self._od = OrderedDict()
        self.hits = 0
        self.misses = 0
    def get(self, key):
        if key in self._od:
            val = self._od.pop(key)
            self._od[key] = val
            self.hits += 1
            return val
        self.misses += 1
        return None
    def put(self, key, value):
        if key in self._od:
            self._od.pop(key)
        self._od[key] = value
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)
    def stats(self):
        return {'size': len(self._od), 'hits': self.hits, 'misses': self.misses}

def hash_state_vec(state_vec):
    v = np.asarray(state_vec, dtype=np.float32)
    # coarse quantization for cache robustness
    q = np.round(v * 100.0) / 100.0
    return tuple(q.tolist())

def _sim_worker(conn, table, balls, action, my_targets):
    try:
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(V0=action['V0'], phi=action['phi'], theta=action['theta'], a=action['a'], b=action['b'])
        pt.simulate(shot, inplace=True)
        last_state_snapshot = {bid: copy.deepcopy(balls[bid]) for bid in balls.keys()}
        reward = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
        conn.send({'ok': True, 'reward': float(reward)})
    except Exception as e:
        conn.send({'ok': False, 'error': str(e)})
    finally:
        conn.close()

def simulate_with_timeout_crossplat(table, balls, action, my_targets, timeout=1.0):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_sim_worker, args=(child_conn, table, balls, action, my_targets))
    p.start()
    if parent_conn.poll(timeout):
        msg = parent_conn.recv()
        p.join()
        if msg.get('ok'):
            return True, float(msg.get('reward', 0.0))
        return False, -500.0
    try:
        p.terminate()
    except Exception:
        pass
    return False, -500.0

def select_top_k_by_proxy(state_vec, actions, K):
    scores = [proxy_score(state_vec, a, []) for a in actions]
    idx = np.argsort(scores)[::-1][:K]
    return [actions[int(i)] for i in idx], [scores[int(i)] for i in idx]

def pipeline_best_action(balls, my_targets, table, candidate_actions, K, cache: LRUCache, timeout=1.0):
    state_vec = encode_state(balls, my_targets, table)
    top_actions, _ = select_top_k_by_proxy(state_vec, candidate_actions, K)
    best = None
    best_r = -1e9
    hstate = hash_state_vec(state_vec)
    for a in top_actions:
        key = (hstate, round(a['V0'], 2), round(a['phi'], 1), round(a['theta'], 1), round(a['a'], 2), round(a['b'], 2))
        cached = cache.get(key)
        if cached is not None:
            r = float(cached)
        else:
            ok, r = simulate_with_timeout_crossplat(table, balls, a, my_targets, timeout=timeout)
            cache.put(key, r)
        if r > best_r:
            best_r = r
            best = a
    return best, best_r
