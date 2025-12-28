import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
from .utils import get_straight_in_all
import copy
import signal
import random

from .agent import Agent

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=10):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        pass
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        
        if balls is None:
            print(f"[NewAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        
        # 保存击球前的状态快照，用于评分对比
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 更新目标球（如果自己的球已全部清空，切换到8号球）
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_own) == 0:
            my_targets = ["8"]
            print("[NewAgent] 我的目标球已全部清空，自动切换目标为：8号球")

        # 进攻：获取所有可能的直线进球方案
        possiblities = get_straight_in_all(balls, my_targets, table)
        print(f"[NewAgent] 目标球: {my_targets}, 找到 {len(possiblities)} 个可能的进攻方案")
        
        # 按难度排序（难度越低越优先）
        ranked_possiblties = sorted(possiblities, key=lambda x: x['difficulty'])
        
        # 对每个可能的进攻动作进行实际模拟评估
        for i, possibility in enumerate(ranked_possiblties):
            if possibility['difficulty'] >= 100:
                # 难度太高，跳过后续所有方案（已按难度排序）
                print(f"[NewAgent] 剩余方案难度均 >= 100，停止评估")
                break
            
            action = possibility['action']
            
            # 模拟此动作的实际效果
            try:
                # 创建模拟环境（深拷贝避免影响真实状态）
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                # 设置球杆参数
                shot.cue.set_state(
                    V0=action['V0'],
                    phi=action['phi'],
                    theta=action['theta'],
                    a=action['a'],
                    b=action['b']
                )
                
                # 使用带超时保护的物理模拟（10秒上限）
                if not simulate_with_timeout(shot, timeout=10):
                    print(f"[NewAgent] 方案 {i+1}: 模拟超时，跳过 (球 {possibility['target']} -> 袋 {possibility['pocket']})")
                    continue
                
                # 使用规则引擎评分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                print(f"[NewAgent] 方案 {i+1}: 球 {possibility['target']} -> 袋 {possibility['pocket']}, "
                      f"难度={possibility['difficulty']:.2f}, 模拟得分={score:.2f}")
                
                # 判断得分是否理想（>= 40 表示至少进一球且无重大犯规）
                if score >= 40:
                    print(f"[NewAgent] ✓ 选择此方案执行 (得分: {score:.2f})")
                    return action
                    
            except Exception as e:
                print(f"[NewAgent] 方案 {i+1}: 模拟失败，跳过。原因: {e}")
                continue
        
        # 防守：如果所有进攻方案都不理想，执行保守策略
        print("[NewAgent] 未找到理想的进攻方案（得分 < 40)执行防守动作")

        count = 0

        while True:
            r_phi = 5 * random.randint(0, 71)
            r_v0 = 1.0 + 0.5 * random.randint(0, 2)

            count += 1
            if (count > 150):
                break

            try:
                # 创建模拟环境（深拷贝避免影响真实状态）
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                # 设置球杆参数
                shot.cue.set_state(
                    V0=r_v0,
                    phi=r_phi,
                    theta=5,
                    a=0,
                    b=0
                )
                
                # 使用带超时保护的物理模拟（10秒上限）
                if not simulate_with_timeout(shot, timeout=10):
                    print(f"模拟超时，跳过")
                    continue
                
                # 使用规则引擎评分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                # 判断防守是否理想（>= 0 表示无重大犯规）
                if score >= 0:
                    print(f"[NewAgent] ✓ 选择此方案执行 (得分: {score:.2f})")
                    return {"V0":r_v0, "phi":r_phi, "theta":5, "a":0, "b":0}
                    
            except Exception as e:
                print(f"[NewAgent] 方案 {i+1}: 模拟失败，跳过。原因: {e}")
                continue

        print("Choose random shot instead.")
        return self._random_action()