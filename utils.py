import random
import numpy as np
import math

try:
    import torch
except ImportError:
    torch = None

ball_radius = 0.02625
g = 9.81
mass = 0.1406  # kg
restitution = 0.98  # 碰撞恢复系数
table_friction = 0.212

def set_random_seed(enable=False, seed=42):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        enable (bool): 是否启用固定随机种子
        seed (int): 当 enable 为 True 时使用的随机种子
    """
    if enable:
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        
        # 设置 PyTorch 随机种子（如果可用）
        if torch is not None:
            torch.manual_seed(seed)  # CPU 随机种子
            torch.cuda.manual_seed(seed)  # 当前 GPU 随机种子
            torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"随机种子已设置为: {seed}")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        np.random.seed(None)
        
        print("随机种子已禁用，使用完全随机模式")

def get_straight_in_all(balls=None, my_targets=None, table=None):
    """
    get_straight_in_all 的 Docstring
    
    :param balls: 
    :param my_targets: 
    :param table: 

    找所有直接一次碰撞进袋的路线
    """
    decision_list = []

    for target in my_targets:
        if balls[target].state.s == 0:
            for pocket_id, pocket in table.pockets.items():
                decision_list.append(find_straight_in_way(balls, my_targets, table, target, pocket.center, pocket_id))

    if decision_list == []:
        for pocket_id, pocket in table.pockets.items():
                decision_list.append(find_straight_in_way(balls, my_targets, table, '8', pocket.center, pocket_id))

    return decision_list

def dist_between(a, b):
    a = np.asarray(a)[:2]
    b = np.asarray(b)[:2]
    return float(np.linalg.norm(a - b))

def cross_cos(cue_pos, target_pos_2d, target_ball_pos_2d, pocket_2d):
    target_to_pocket = dist_between(target_ball_pos_2d, pocket_2d)

    # cue->target 方向
    cue_to_target_vec = target_pos_2d - cue_pos
    cue_to_target_vec = cue_to_target_vec / np.linalg.norm(cue_to_target_vec)
    # target->pocket 方向
    target_to_pocket_vec = pocket_2d - target_ball_pos_2d
    target_to_pocket_vec = target_to_pocket_vec / np.linalg.norm(target_to_pocket_vec)
    # 夹角余弦
    cos_alpha = np.dot(cue_to_target_vec, target_to_pocket_vec)
    cos_alpha = np.clip(cos_alpha, 0.01, 1.0)  # 防止极端情况

    return cos_alpha

def min_v0(cue_pos, target_pos_2d, target_ball_pos_2d, pocket_2d):

    # 计算触球角对能量传递的影响

    target_to_pocket = dist_between(target_ball_pos_2d, pocket_2d)

    # cue->target 方向
    cue_to_target_vec = target_pos_2d - cue_pos
    cue_to_target_vec = cue_to_target_vec / np.linalg.norm(cue_to_target_vec)
    # target->pocket 方向
    target_to_pocket_vec = pocket_2d - target_ball_pos_2d
    target_to_pocket_vec = target_to_pocket_vec / np.linalg.norm(target_to_pocket_vec)
    # 夹角余弦
    cos_alpha = np.dot(cue_to_target_vec, target_to_pocket_vec)
    cos_alpha = np.clip(cos_alpha, 0.01, 1.0)  # 防止极端情况

    # 目标球需要的速度
    v_target = math.sqrt(2 * table_friction * g * target_to_pocket)
    # 实际碰撞时母球需要给目标球的速度（修正能量传递效率）
    v_cue_after = v_target / (restitution * cos_alpha)
    cue_to_target = dist_between(cue_pos, target_pos_2d)
    v0_th = v_cue_after + math.sqrt(2 * table_friction * g * cue_to_target)

    return v0_th

def find_straight_in_way(balls, my_targets, table, target, pocket_center, pocket_id):

    target_ball_pos = balls[target].state.rvw[0]
    dist_targetball_to_pocket = dist_between(target_ball_pos, pocket_center)
    target_pos = target_ball_pos + (target_ball_pos - pocket_center) * 2 * ball_radius / dist_targetball_to_pocket
    dist_cue_to_target = dist_between(balls['cue'].state.rvw[0], target_pos)

    target_pos[2] = target_ball_pos[2] # z remains

    trace_cue_to_target = target_pos - balls['cue'].state.rvw[0]
    trace_targetball_to_pocket = pocket_center - target_ball_pos

    action = {}
    action['a'] = 0
    action['b'] = 0
    action['phi'] = math.degrees(math.atan2(trace_cue_to_target[1], trace_cue_to_target[0])) % 360

    action['theta'] = 5

    v0_ratio = 0.2
    v0_th = min_v0(balls['cue'].state.rvw[0][:2], target_pos[:2], target_ball_pos[:2], pocket_center[:2]) * v0_ratio
    v0_pr = 2
    while (v0_pr <= v0_th):
        v0_pr += 0.5
        if v0_pr >= 8:
            break
    action['V0'] = v0_pr

    difficulty = 0

    # 球路被阻挡 cue_to_target

    hit_message = ""
    hit_ratio = 0.02
    for ball_id, ball in balls.items():
        if (ball.state.s == 0) and (ball_id != 'cue'):
            if hit_in_trace(ball.state.rvw[0][:2], (target_pos - hit_ratio * trace_cue_to_target)[:2], balls['cue'].state.rvw[0][:2]):
                hit_message = f"When try to hit ball {target} cue ball hit ball {ball_id} midway."
                difficulty = 100
                break
    
    # 球路被阻挡 targetball_to_pocket

    for ball_id, ball in balls.items():
        if (ball.state.s == 0) and (ball_id != target):
            if hit_in_trace(ball.state.rvw[0][:2], (pocket_center - hit_ratio * trace_targetball_to_pocket)[:2], target_ball_pos[:2]):
                difficulty = 100
                break

    # 目标位置不合法
    if (target_pos[0] < ball_radius) or (target_pos[1] < ball_radius) or (target_pos[0] + ball_radius > table.w) or (target_pos[1] + ball_radius > table.l):
        difficulty = 100

    # difficulty和dist正相关，和击球角的余弦值正相关
    # Limit: dist_total = 1.5, cos = 0.7 此时大概difficulty = 1

    if (difficulty == 0):
        beta = 3
        dist_ratio = 1.5
        angle_ratio = 1.3
        difficulty = (dist_targetball_to_pocket + dist_cue_to_target) / dist_ratio
        diificulty = difficulty * (1 - angle_ratio * abs(cross_cos(balls['cue'].state.rvw[0][:2], target_pos[:2], target_ball_pos[:2], pocket_center[:2])))

        # 中袋角度修正
        if (pocket_id in ['lc', 'rc']):
            difficulty = difficulty / (1 - abs(trace_targetball_to_pocket[1] / math.sqrt(trace_targetball_to_pocket[0] ** 2 + trace_targetball_to_pocket[1] ** 2)))
        if (difficulty > 100):
            difficulty = 100

    return {'action':action, 'difficulty':difficulty, 'target':target, 'hit_message':hit_message, 'pocket':pocket_id}

def hit_in_trace(hit_ball, dest_pos, start_pos):

    if dist_between(hit_ball, dest_pos) < 2 * ball_radius: 
        return True
    
    seg = np.array(dest_pos) - np.array(start_pos)
    seg_len = np.linalg.norm(seg)
    seg_dir = seg / seg_len
    # 2. 向量 from start to ball
    to_ball = np.array(hit_ball) - np.array(start_pos)
    # 3. 投影长度
    proj = np.dot(to_ball, seg_dir)
    # 4. 判断投影是否在线段内
    if proj < 0 or proj > seg_len:
        return False
    # 5. 计算垂直距离
    closest = np.array(start_pos) + proj * seg_dir
    dist = np.linalg.norm(np.array(hit_ball) - closest)
    # print(f"dist:{dist}")
    return dist < 2 * ball_radius
    



