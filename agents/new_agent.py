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
    """带超时保护的物理模拟（SIGALRM：Unix/Linux）"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)
# ============================================


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    （保持你原逻辑不动）
    """
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

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

    score = 0

    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 100
        else:
            score -= 150

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    return score


class NewAgent(Agent):
    """
    均衡版 NewAgent（胜率↑，速度损失不大）：
    - 候选：straight-in + cut(ghost-ball)
    - 进攻：只仿真少量 TopK（默认 6）
    - eval：规则分 +（必要时）局面分（连攻/不送球）+ 清台进度 - 白球近袋风险
    - 防守：不再 0..360 全扫描；改为“定向少量候选 + 少量随机扰动”，仿真 ~18-27 次
    """

    def __init__(self):
        super().__init__()

        # ---------- 仿真 ----------
        self.SIM_TIMEOUT = 10

        # ---------- 进攻搜索规模（决定速度的关键） ----------
        self.TOPK_ATTACK = 6              # ✅ 少量仿真
        self.DIFFICULTY_MARGIN = 20.0     # ✅ 更窄窗口减少候选

        # ---------- 局面评估权重 ----------
        self.W_MY_NEXT = 0.8
        self.W_OPP_NEXT = 1.6
        self.MY_EASY_REF = 85.0
        self.OPP_EASY_REF = 65.0

        self.W_CLEAR_PROGRESS = 18.0
        self.W_CUE_NEAR_POCKET = 25.0     # 略降，避免过度牺牲进攻走位

        # ---------- 剪枝：哪些情况下才计算“下一杆机会”（省时） ----------
        self.EVAL_LOOKAHEAD_BASE_TH = 20  # base>=20 才算局面项（可调：越大越快）
        self.ATTACK_MIN_ACCEPT = 25.0

        # ---------- 防守：少量定向候选（很省时） ----------
        self.DEFENSE_N_DIR = 7            # 生成7个方向
        self.DEFENSE_V_CHOICES = [0.9, 1.4, 1.9]  # 3档力度
        self.DEFENSE_THETA = 5.0
        self.DEFENSE_JITTER_DEG = 6.0     # 每个方向加扰动，避免死板

        # ---------- 几何近似 ----------
        self.BALL_R = 0.028575
        self.CLEARANCE = 0.002
        self.CUE_NEAR_POCKET_DIST = 0.10

    # ------------------ 基础工具：位置/袋口 ------------------
    def _pos2(self, ball):
        st = getattr(ball, "state", ball)
        for key in ("r", "xyz", "pos", "position", "p"):
            r = getattr(st, key, None)
            if r is not None and hasattr(r, "__len__") and len(r) >= 2:
                return np.array([float(r[0]), float(r[1])], dtype=np.float64)
        x = getattr(st, "x", None)
        y = getattr(st, "y", None)
        if x is not None and y is not None:
            return np.array([float(x), float(y)], dtype=np.float64)
        raise AttributeError("Cannot find ball position fields (state.r/xyz/pos/position/p or state.x/y).")

    def _get_pockets2(self, table):
        candidates = []
        if hasattr(table, "pockets"):
            candidates.append(getattr(table, "pockets"))
        if hasattr(table, "specs") and hasattr(table.specs, "pockets"):
            candidates.append(table.specs.pockets)

        pockets_xy = []
        for pockets in candidates:
            if pockets is None:
                continue
            try:
                for pk in pockets:
                    if hasattr(pk, "x") and hasattr(pk, "y"):
                        pockets_xy.append(np.array([float(pk.x), float(pk.y)], dtype=np.float64))
                        continue
                    for key in ("r", "pos", "center", "xyz", "position"):
                        rr = getattr(pk, key, None)
                        if rr is not None and hasattr(rr, "__len__") and len(rr) >= 2:
                            pockets_xy.append(np.array([float(rr[0]), float(rr[1])], dtype=np.float64))
                            break
                    else:
                        if isinstance(pk, (list, tuple)) and len(pk) >= 2:
                            pockets_xy.append(np.array([float(pk[0]), float(pk[1])], dtype=np.float64))
            except Exception:
                continue

        if pockets_xy:
            return pockets_xy
        raise AttributeError("Cannot find pocket positions (table.pockets or table.specs.pockets).")

    def _remaining_targets(self, balls: dict, targets: list):
        return [bid for bid in targets if bid in balls and balls[bid].state.s != 4]

    def _count_remaining(self, balls: dict, targets: list):
        return len(self._remaining_targets(balls, targets))

    def _infer_opp_targets(self, balls: dict, my_targets: list):
        solids = [str(i) for i in range(1, 8)]
        stripes = [str(i) for i in range(9, 16)]

        if my_targets == ['8']:
            alive = [bid for bid, b in balls.items() if bid not in ['cue'] and b.state.s != 4]
            opp = [bid for bid in alive if bid != '8']
            return opp if opp else ['8']

        s_my = set(my_targets)
        if s_my.issubset(set(solids)):
            return solids
        if s_my.issubset(set(stripes)):
            return stripes
        return [str(i) for i in range(1, 16) if str(i) not in my_targets]

    def _min_difficulty(self, possibilities: list):
        if not possibilities:
            return 1e9
        return min(float(p.get("difficulty", 1e9)) for p in possibilities)

    def _easy_score(self, min_diff: float, ref: float):
        return max(0.0, ref - float(min_diff))

    def _cue_near_pocket_penalty(self, balls_after: dict, table):
        try:
            cue = balls_after.get("cue", None)
            if cue is None or cue.state.s == 4:
                return 0.0
            cue_p = self._pos2(cue)
            pockets = self._get_pockets2(table)
            dmin = min(float(np.linalg.norm(cue_p - pk)) for pk in pockets)
            if dmin < self.CUE_NEAR_POCKET_DIST:
                return self.W_CUE_NEAR_POCKET * (self.CUE_NEAR_POCKET_DIST - dmin) / max(1e-6, self.CUE_NEAR_POCKET_DIST)
            return 0.0
        except Exception:
            return 0.0

    def _segment_clear(self, p0, p1, balls, ignore_ids=set(), clearance=0.0):
        v = p1 - p0
        L2 = float(v @ v)
        if L2 < 1e-10:
            return False
        rad = 2.0 * self.BALL_R + clearance

        for bid, b in balls.items():
            if bid in ignore_ids or b.state.s == 4:
                continue
            c = self._pos2(b)
            t = float(((c - p0) @ v) / L2)
            if t <= 0.0 or t >= 1.0:
                continue
            closest = p0 + t * v
            if float(np.linalg.norm(c - closest)) < rad:
                return False
        return True

    # ------------------ 切球候选（ghost-ball） ------------------
    def _gen_cut_candidates(self, balls, my_targets, table):
        out = []
        try:
            pockets = self._get_pockets2(table)
            cue_p = self._pos2(balls["cue"])
        except Exception:
            return out

        for tid in my_targets:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = self._pos2(balls[tid])

            for pi, pk in enumerate(pockets):
                d = pk - tpos
                n = float(np.linalg.norm(d))
                if n < 1e-6:
                    continue
                u = d / n
                ghost = tpos - u * (2.0 * self.BALL_R)

                if not self._segment_clear(tpos, pk, balls, ignore_ids={tid, "cue"}, clearance=self.CLEARANCE):
                    continue
                if not self._segment_clear(cue_p, ghost, balls, ignore_ids={tid, "cue"}, clearance=self.CLEARANCE):
                    continue

                aim = ghost - cue_p
                dist = float(np.linalg.norm(aim))
                if dist < 1e-6:
                    continue

                phi = (math.degrees(math.atan2(aim[1], aim[0])) + 360.0) % 360.0
                V0 = float(np.clip(1.0 + 0.65 * dist, 0.8, 3.0))

                dp = float(np.linalg.norm(pk - tpos))
                difficulty = 18.0 + 7.0 * dist + 2.5 * dp  # 略“乐观”，让它更愿意尝试切球

                out.append({
                    "difficulty": difficulty,
                    "action": {"V0": V0, "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0},
                    "target": tid,
                    "pocket": pi,
                    "type": "cut"
                })
        return out

    # ------------------ 仿真封装 ------------------
    def _simulate_action(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        shot.cue.set_state(
            V0=float(action['V0']),
            phi=float(action['phi']),
            theta=float(action['theta']),
            a=float(action['a']),
            b=float(action['b'])
        )
        ok = simulate_with_timeout(shot, timeout=self.SIM_TIMEOUT)
        return ok, shot

    # ------------------ 评估（带剪枝） ------------------
    def _evaluate_shot(self, shot: pt.System, last_state_snapshot: dict, my_targets: list, my_before_left: int):
        base = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)

        # 极差直接返回（省时）
        if base <= -120:
            return base

        after_balls = shot.balls

        my_left_after = self._count_remaining(after_balls, my_targets)
        my_after_targets = ['8'] if (my_targets != ['8'] and my_left_after == 0) else my_targets

        # 清台进度：很便宜但很有效
        progress = float(my_before_left - my_left_after)
        prog_term = self.W_CLEAR_PROGRESS * progress

        # 白球近袋风险：便宜
        cue_pen = self._cue_near_pocket_penalty(after_balls, shot.table)

        # ✅ 剪枝：只有 base 足够好才计算“下一杆机会”（贵）
        if base < self.EVAL_LOOKAHEAD_BASE_TH:
            return base + prog_term - cue_pen

        opp_targets = self._infer_opp_targets(after_balls, my_after_targets)

        try:
            my_next = get_straight_in_all(after_balls, my_after_targets, shot.table)
            opp_next = get_straight_in_all(after_balls, opp_targets, shot.table)
            my_min_d = self._min_difficulty(my_next)
            opp_min_d = self._min_difficulty(opp_next)
        except Exception:
            my_min_d = 1e9
            opp_min_d = 1e9

        my_term = self.W_MY_NEXT * self._easy_score(my_min_d, self.MY_EASY_REF)
        opp_term = self.W_OPP_NEXT * self._easy_score(opp_min_d, self.OPP_EASY_REF)

        return base + my_term - opp_term + prog_term - cue_pen

    # ------------------ 防守：定向少量候选 ------------------
    def _gen_defense_dirs(self, balls, table):
        """
        生成少量“有目的”的防守方向（phi 列表）：
        - 指向每个袋口（把白球推离袋口/控制路线）
        - 指向几个随机方向（兜底）
        - 指向球群中心（制造拥堵）
        """
        dirs = []
        try:
            cue_p = self._pos2(balls["cue"])
            pockets = self._get_pockets2(table)
        except Exception:
            return [float(5 * random.randint(0, 71)) for _ in range(self.DEFENSE_N_DIR)]

        # 1) 指向袋口的反方向（让白球远离袋口更常见）
        # 用“cue -> pocket”的方向，加180度当作“远离袋口”
        for pk in pockets[:4]:  # 取前4个袋口就够了（省时）
            v = pk - cue_p
            if float(np.linalg.norm(v)) < 1e-6:
                continue
            phi_to = (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0
            dirs.append((phi_to + 180.0) % 360.0)

        # 2) 指向球群中心（制造复杂局面）
        pts = []
        for bid, b in balls.items():
            if bid in ("cue",) or b.state.s == 4:
                continue
            try:
                pts.append(self._pos2(b))
            except Exception:
                pass
        if pts:
            center = np.mean(np.stack(pts, axis=0), axis=0)
            v = center - cue_p
            if float(np.linalg.norm(v)) > 1e-6:
                dirs.append((math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0)

        # 3) 少量随机兜底
        while len(dirs) < self.DEFENSE_N_DIR:
            dirs.append(float(5 * random.randint(0, 71)))

        # 去重+截断
        uniq = []
        for d in dirs:
            if all(abs(((d - u + 180) % 360) - 180) > 8 for u in uniq):
                uniq.append(d)
        return uniq[:self.DEFENSE_N_DIR]

    def _defense_action(self, balls, my_targets, table, last_state_snapshot, my_before_left):
        best_action = None
        best_score = -1e18

        dirs = self._gen_defense_dirs(balls, table)

        for base_phi in dirs:
            for _ in range(2):  # 每个方向两次扰动
                phi = float((base_phi + random.uniform(-self.DEFENSE_JITTER_DEG, self.DEFENSE_JITTER_DEG)) % 360.0)
                for V0 in self.DEFENSE_V_CHOICES:
                    action = {"V0": float(V0), "phi": phi, "theta": float(self.DEFENSE_THETA), "a": 0.0, "b": 0.0}
                    try:
                        ok, shot = self._simulate_action(balls, table, action)
                        if not ok:
                            continue
                        eval_score = self._evaluate_shot(shot, last_state_snapshot, my_targets, my_before_left)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_action = action
                    except Exception:
                        continue

        if best_action is not None:
            print(f"[NewAgent] ✓ DEFENSE eval={best_score:.2f} V0={best_action['V0']:.2f} phi={best_action['phi']:.1f}")
            return best_action

        return self._random_action()

    # ------------------ 主决策 ------------------
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 清台 -> 打8
        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_own) == 0:
            my_targets = ["8"]

        my_before_left = self._count_remaining(balls, my_targets)

        # 进攻候选：straight + cut
        try:
            straight_poss = get_straight_in_all(balls, my_targets, table)
        except Exception:
            straight_poss = []
        cut_poss = self._gen_cut_candidates(balls, my_targets, table)
        all_poss = list(straight_poss) + list(cut_poss)

        if not all_poss:
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        ranked = sorted(all_poss, key=lambda x: float(x.get("difficulty", 1e9)))
        min_d = float(ranked[0].get("difficulty", 1e9))
        filtered = [p for p in ranked if float(p.get("difficulty", 1e9)) <= (min_d + self.DIFFICULTY_MARGIN)]
        candidates = filtered[:self.TOPK_ATTACK]

        best_action = None
        best_score = -1e18

        for p in candidates:
            action = p["action"]
            try:
                ok, shot = self._simulate_action(balls, table, action)
                if not ok:
                    continue
                eval_score = self._evaluate_shot(shot, last_state_snapshot, my_targets, my_before_left)
                if eval_score > best_score:
                    best_score = eval_score
                    best_action = action
            except Exception:
                continue

        if best_action is not None and best_score >= self.ATTACK_MIN_ACCEPT:
            return best_action

        return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)
