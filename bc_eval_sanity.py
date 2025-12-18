import argparse
import numpy as np
import os
import torch
import pooltool as pt
from poolenv import PoolEnv
from agent import NewAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--noise', type=int, default=0)
    args = parser.parse_args()
    env = PoolEnv()
    if args.noise == 0:
        env.enable_noise = False
    agent = NewAgent()
    pockets = 0
    fouls = 0
    shots = 0
    for i in range(args.games):
        env.reset(target_ball=['solid','solid','stripe','stripe'][i % 4])
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            action = agent.decision(balls, my_targets, table)
            info = env.take_shot(action)
            shots += 1
            pockets += len(info.get('ME_INTO_POCKET', []))
            fouls += int(info.get('WHITE_BALL_INTO_POCKET', False)) + int(info.get('FOUL_FIRST_HIT', False)) + int(info.get('NO_POCKET_NO_RAIL', False)) + int(info.get('NO_HIT', False))
            if args.render > 0 and shots <= args.render:
                pt.show(env.shot_record[-1], title=f"shot {shots}")
            done, _ = env.get_done()
            if done or shots >= args.games * 20:
                break
    print(f"[BC sanity] shots={shots} pockets={pockets} fouls={fouls}")

if __name__ == '__main__':
    main()
