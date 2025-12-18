import argparse
import os
import numpy as np
import random
from poolenv import PoolEnv
from agent import BasicAgent
from utils_state import encode_state, action_norm, load_meta
import json

def compute_reward_from_step(step_info, my_targets):
    score = 0.0
    if step_info.get('WHITE_BALL_INTO_POCKET', False) and step_info.get('BLACK_BALL_INTO_POCKET', False):
        return -150.0
    if step_info.get('WHITE_BALL_INTO_POCKET', False):
        score -= 100.0
    if step_info.get('BLACK_BALL_INTO_POCKET', False):
        legal = (len(my_targets) == 1 and my_targets[0] == '8')
        score += 100.0 if legal else -150.0
    if step_info.get('FOUL_FIRST_HIT', False):
        score -= 30.0
    if step_info.get('NO_POCKET_NO_RAIL', False):
        score -= 30.0
    own = step_info.get('ME_INTO_POCKET', [])
    enemy = step_info.get('ENEMY_INTO_POCKET', [])
    score += float(len(own) * 50 - len(enemy) * 20)
    if score == 0.0 and not step_info.get('WHITE_BALL_INTO_POCKET', False) and not step_info.get('BLACK_BALL_INTO_POCKET', False) and not step_info.get('FOUL_FIRST_HIT', False) and not step_info.get('NO_POCKET_NO_RAIL', False):
        score = 10.0
    return score

def run_games(n_games, target_cycle, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = PoolEnv()
    expert = BasicAgent()
    states = []
    actions = []
    rewards = []
    dones = []
    next_states = []
    meta = load_meta(None, 2 + 15*4 + 2 + 6*2)
    stats = {
        'me_pocket': 0,
        'enemy_pocket': 0,
        'white_pocket': 0,
        'black_pocket_legal': 0,
        'black_pocket_illegal': 0,
        'foul_first_hit': 0,
        'no_pocket_no_rail': 0,
        'no_hit': 0,
        'steps': 0
    }
    for i in range(n_games):
        env.reset(target_ball=target_cycle[i % len(target_cycle)])
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            s = encode_state(balls, my_targets, table)
            a = expert.decision(balls, my_targets, table)
            step_info = env.take_shot(a)
            r = compute_reward_from_step(step_info, my_targets)
            done, info = env.get_done()
            balls2 = step_info['BALLS']
            s_next = encode_state(balls2, my_targets, table)
            states.append(s)
            actions.append(action_norm(a, meta))
            rewards.append(r)
            dones.append(float(1.0 if done else 0.0))
            next_states.append(s_next)
            stats['me_pocket'] += len(step_info.get('ME_INTO_POCKET', []))
            stats['enemy_pocket'] += len(step_info.get('ENEMY_INTO_POCKET', []))
            stats['white_pocket'] += 1 if step_info.get('WHITE_BALL_INTO_POCKET') else 0
            if step_info.get('BLACK_BALL_INTO_POCKET'):
                legal = (len(my_targets) == 1 and my_targets[0] == '8')
                stats['black_pocket_legal'] += 1 if legal else 0
                stats['black_pocket_illegal'] += 0 if legal else 1
            stats['foul_first_hit'] += 1 if step_info.get('FOUL_FIRST_HIT') else 0
            stats['no_pocket_no_rail'] += 1 if step_info.get('NO_POCKET_NO_RAIL') else 0
            stats['no_hit'] += 1 if step_info.get('NO_HIT') else 0
            stats['steps'] += 1
            if done:
                break
    return {
        'states': np.asarray(states, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'dones': np.asarray(dones, dtype=np.float32),
        'next_states': np.asarray(next_states, dtype=np.float32),
        'stats': stats,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/bc_data.npz')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    target_cycle = ['solid', 'solid', 'stripe', 'stripe']
    data = run_games(args.games, target_cycle, args.seed)
    np.savez_compressed(args.out, **data)
    stats_path = os.path.join(os.path.dirname(args.out), 'bc_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(data['stats'], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
