import os
import argparse
import numpy as np
from poolenv import PoolEnv
from agent import BasicAgent
from utils_state import encode_state
from sim_pipeline import LRUCache, pipeline_best_action

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--cache', type=int, default=20000)
    parser.add_argument('--timeout', type=float, default=1.0)
    args = parser.parse_args()
    env = PoolEnv()
    agent = BasicAgent()
    cache = LRUCache(args.cache)
    target_cycle = ['solid', 'solid', 'stripe', 'stripe']
    for i in range(args.games):
        env.reset(target_ball=target_cycle[i % 4])
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            base = agent.decision(balls, my_targets, table)
            grid = []
            for dv in [0.0, -0.3, 0.3]:
                for dphi in [0.0, -5.0, 5.0]:
                    for dth in [0.0, -3.0, 3.0]:
                        grid.append({
                            'V0': float(np.clip(base['V0'] + dv, 0.5, 8.0)),
                            'phi': float((base['phi'] + dphi) % 360.0),
                            'theta': float(np.clip(base['theta'] + dth, 0.0, 90.0)),
                            'a': float(np.clip(base['a'], -0.5, 0.5)),
                            'b': float(np.clip(base['b'], -0.5, 0.5)),
                        })
            best_a, _ = pipeline_best_action(balls, my_targets, table, grid, K=args.k, cache=cache, timeout=args.timeout)
            env.take_shot(best_a if best_a is not None else base)
            done, _ = env.get_done()
            if done:
                break
    print("cache:", cache.stats())

if __name__ == '__main__':
    main()
