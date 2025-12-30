import argparse
from agents.utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgent, BasicAgentPro, NewAgent


def build_agent_a(name: str):
    name = name.lower()
    if name in ("basic", "basica", "basicagent"):
        return BasicAgent()
    if name in ("pro", "basicpro", "basicagentpro"):
        return BasicAgentPro()
    raise ValueError(f"Unknown --agent-a: {name}. Use: basic | pro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-a",
        type=str,
        default="pro",
        help="Agent A type: basic | pro (default: pro)"
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=20,
        help="Number of games (default: 20; grading often uses 120)"
    )
    parser.add_argument(
        "--seed-enable",
        action="store_true",
        help="Enable fixed random seed for reproducibility"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value when --seed-enable is set (default: 42)"
    )
    args = parser.parse_args()

    # Agent B 永远是 NewAgent（默认且不可改）
    agent_a = build_agent_a(args.agent_a)
    agent_b = NewAgent()

    set_random_seed(enable=args.seed_enable, seed=args.seed)

    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

    players = [agent_a, agent_b]  # 用于切换先后手

    for i in range(args.n_games):
        print()
        print(f"------- 第 {i} 局比赛开始 -------")
        env.reset(target_ball=target_ball_choice[i % 4])

        first_player = players[i % 2]
        second_player = players[(i + 1) % 2]

        player_class = first_player.__class__.__name__
        ball_type = target_ball_choice[i % 4]
        print(f"本局 Player A(先手): {player_class}, 目标球型: {ball_type}")

        while True:
            player = env.get_curr_player()
            print(f"[第{env.hit_count}次击球] player: {player}")
            obs = env.get_observation(player)

            action = first_player.decision(*obs) if player == 'A' else second_player.decision(*obs)
            step_info = env.take_shot(action)

            done, info = env.get_done()
            if not done:
                if step_info.get('ENEMY_INTO_POCKET'):
                    print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
            else:
                if info['winner'] == 'SAME':
                    results['SAME'] += 1
                elif info['winner'] == 'A':
                    # 对局内 A 胜：first_player 胜
                    if i % 2 == 0:
                        results['AGENT_A_WIN'] += 1
                    else:
                        results['AGENT_B_WIN'] += 1
                else:  # info['winner'] == 'B'
                    # 对局内 B 胜：second_player 胜
                    if i % 2 == 0:
                        results['AGENT_B_WIN'] += 1
                    else:
                        results['AGENT_A_WIN'] += 1
                break

    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] + results['SAME'] * 0.5

    print("\n最终结果：", results)
    print(f"AgentA={agent_a.__class__.__name__}, AgentB=NewAgent, n_games={args.n_games}, "
          f"seed_enable={args.seed_enable}, seed={args.seed}")


if __name__ == "__main__":
    main()
