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
    强化版（冲90%+思路）：
    1) 候选动作结构化生成：straight + cut(ghost-ball) + 同一动作多力度
    2) 先快评估（几何/覆盖率/deny）筛TopK，再物理仿真精评（少量）
    3) eval 从 min_d 升级为 Top-3 聚合 + cue 停位覆盖率 + deny + 清台进度 + 风险惩罚
    4) 防守用定向少量候选（deny/hide），不做全角扫描
    """

    def __init__(self):
        super().__init__()

        # ---- 性能/搜索规模 ----
        self.SIM_TIMEOUT = 10
        self.ATTACK_PREFILTER_K = 18      # 快评估后保留多少进入仿真
        self.ATTACK_SIM_K = 7             # 最终物理仿真多少个（核心提速点）
        self.CUT_MAX_PER_TARGET = 3       # 每个目标最多保留多少个切球候选（防爆炸）

        # ---- 动作力度档（同一方向三档）----
        # 会把候选动作 V0 乘以这些系数，来控制走位
        self.POWER_SCALES_ATTACK = [0.85, 1.0, 1.18]
        self.POWER_SCALES_DEF = [0.85, 1.1, 1.35]

        # ---- 几何近似参数 ----
        self.BALL_R = 0.028575
        self.CLEARANCE = 0.002
        self.CUE_NEAR_POCKET_DIST = 0.10

        # ---- 价值函数权重（偏向连攻+deny） ----
        self.W_RULE = 1.0
        self.W_MY_TOP3 = 0.55            # 我方Top3机会（连攻）
        self.W_OPP_TOP3 = 0.95           # 对手Top3机会（deny）权重大一些
        self.W_CUE_COVER = 0.85          # 白球停位覆盖率
        self.W_CLEAR_PROGRESS = 18.0     # 清台进度
        self.W_CUE_NEAR_POCKET = 25.0    # 白球近袋惩罚
        self.W_SCRATCH_SOFT = 25.0       # 白球危险区域软惩罚（非进袋，但很危险）

        # Top-3 聚合超参数
        self.EASY_REF_MY = 90.0
        self.EASY_REF_OPP = 70.0
        self.TOP3_ALPHA = 0.55           # e1 + a e2 + a^2 e3

        # ---- 剪枝阈值 ----
        self.ATTACK_MIN_ACCEPT = 25.0    # 最终最佳进攻不够好就转防守
        self.LOOKAHEAD_BASE_TH = 15.0    # base reward >= 15 才做较重的局面评估（省时）

        # ---- 防守候选 ----
        self.DEF_DIR_N = 8               # 方向数
        self.DEF_JITTER = 7.0            # 方向扰动
        self.DEF_V_CHOICES = [0.9, 1.4, 1.9]  # 防守力度基准

    # ------------------ 位置/袋口工具 ------------------
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

    # ------------------ 基础集合 ------------------
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

    # ------------------ 几何遮挡检查 ------------------
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

    # ------------------ Top-3 机会聚合（run-out potential） ------------------
    def _top3_easy_score(self, possibilities: list, ref: float):
        """
        possibilities: get_straight_in_all 返回的列表（每个含 difficulty）
        e_i = max(0, ref - d_i), 聚合 e1 + a e2 + a^2 e3
        """
        if not possibilities:
            return 0.0
        ds = sorted([float(p.get("difficulty", 1e9)) for p in possibilities])
        score = 0.0
        a = float(self.TOP3_ALPHA)
        for i in range(min(3, len(ds))):
            e = max(0.0, float(ref) - ds[i])
            score += (a ** i) * e
        return score

    # ------------------ 白球停位覆盖率（ghost-ball 快速评估） ------------------
    def _cue_cover_score(self, balls, targets, table):
        """
        不仿真，仅几何：从当前 cue 位置出发，
        用 ghost-ball 检查可打的 (target, pocket) 数量/质量，作为“下一杆覆盖率”。
        """
        try:
            cue_p = self._pos2(balls["cue"])
            pockets = self._get_pockets2(table)
        except Exception:
            return 0.0

        cover = 0.0
        # 只看前 N 个目标，避免很贵（通常你剩下球不多）
        for tid in targets[:7]:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = self._pos2(balls[tid])

            # 每个球只取“最近的两个袋口”评估（省时）
            dps = [(float(np.linalg.norm(pk - tpos)), pk) for pk in pockets]
            dps.sort(key=lambda x: x[0])
            for _, pk in dps[:2]:
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

                # 越短越好：加一点质量分，而不是纯计数
                dist = float(np.linalg.norm(ghost - cue_p))
                cover += 1.0 / (1.0 + 1.2 * dist)
        return cover

    # ------------------ 白球近袋风险 ------------------
    def _cue_near_pocket_penalty(self, balls_after, table):
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

    def _scratch_soft_penalty(self, balls_after, table):
        """
        更软的“危险区”惩罚：白球非常靠近袋口但没进袋，
        在有扰动/对手回合时很容易出事。
        """
        try:
            cue = balls_after.get("cue", None)
            if cue is None or cue.state.s == 4:
                return 0.0
            cue_p = self._pos2(cue)
            pockets = self._get_pockets2(table)
            dmin = min(float(np.linalg.norm(cue_p - pk)) for pk in pockets)
            if dmin < (0.6 * self.CUE_NEAR_POCKET_DIST):
                return self.W_SCRATCH_SOFT * (0.6 * self.CUE_NEAR_POCKET_DIST - dmin) / max(1e-6, 0.6 * self.CUE_NEAR_POCKET_DIST)
            return 0.0
        except Exception:
            return 0.0

    # ------------------ 候选：切球（ghost-ball） ------------------
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

            scored = []
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
                dp = float(np.linalg.norm(pk - tpos))

                # 这里 difficulty 只是快筛指标，不必太精确
                difficulty = 18.0 + 7.0 * dist + 2.5 * dp
                scored.append((difficulty, {
                    "difficulty": difficulty,
                    "action": {"V0": float(np.clip(1.0 + 0.65 * dist, 0.8, 3.0)),
                               "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0},
                    "target": tid, "pocket": pi, "type": "cut"
                }))

            scored.sort(key=lambda x: x[0])
            for _, item in scored[:self.CUT_MAX_PER_TARGET]:
                out.append(item)

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

    # ------------------ 快评估（不仿真，用几何+原difficulty） ------------------
    def _fast_rank(self, balls, my_targets, table, cand):
        """
        仅用于预筛：越大越好
        - 优先 low difficulty（进球可能性）
        - 同时偏好“打完后容易连攻”的倾向：用当前局面 cue_cover 作轻微偏置
          （严格意义上应该用仿真后的cue_cover，但那就不快了，所以这里只做很轻的bias）
        """
        d = float(cand.get("difficulty", 1e9))
        base = max(0.0, 120.0 - d)  # difficulty 越小越好
        # 当前局面连攻潜力（轻量 bias）
        cover_bias = 3.0 * self._cue_cover_score(balls, my_targets, table)
        return base + cover_bias

    # ------------------ 精评估（仿真后） ------------------
    def _eval_after_sim(self, shot, last_state_snapshot, my_targets, my_before_left):
        base = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
        after_balls = shot.balls

        # 清台进度（便宜但很有效）
        my_left_after = self._count_remaining(after_balls, my_targets)
        progress = float(my_before_left - my_left_after)
        prog_term = self.W_CLEAR_PROGRESS * progress

        # 风险惩罚（便宜）
        near_pocket_pen = self._cue_near_pocket_penalty(after_balls, shot.table)
        scratch_soft = self._scratch_soft_penalty(after_balls, shot.table)

        # 剪枝：base 不够好就不做重评估（省时）
        if base < self.LOOKAHEAD_BASE_TH:
            return self.W_RULE * base + prog_term - near_pocket_pen - scratch_soft

        # 更新targets（清完则打8）
        my_after_targets = ['8'] if (my_targets != ['8'] and my_left_after == 0) else my_targets
        opp_targets = self._infer_opp_targets(after_balls, my_after_targets)

        # Top-3 机会聚合（连攻与deny）
        try:
            my_poss = get_straight_in_all(after_balls, my_after_targets, shot.table)
            opp_poss = get_straight_in_all(after_balls, opp_targets, shot.table)
            my_top3 = self._top3_easy_score(my_poss, self.EASY_REF_MY)
            opp_top3 = self._top3_easy_score(opp_poss, self.EASY_REF_OPP)
        except Exception:
            my_top3, opp_top3 = 0.0, 0.0

        # 白球停位覆盖率：用 ghost-ball 几何快速估计“下一杆可打多少”
        cue_cover = self._cue_cover_score(after_balls, my_after_targets, shot.table)

        return (
            self.W_RULE * base
            + self.W_MY_TOP3 * my_top3
            - self.W_OPP_TOP3 * opp_top3
            + self.W_CUE_COVER * cue_cover
            + prog_term
            - near_pocket_pen
            - scratch_soft
        )

    # ------------------ 防守候选：定向少量（deny/hide） ------------------
    def _gen_defense_dirs(self, balls, table):
        """
        生成少量定向方向：
        - 远离袋口（更安全）
        - 指向球群中心（制造拥堵/挡线）
        - 少量随机兜底
        """
        dirs = []
        try:
            cue_p = self._pos2(balls["cue"])
            pockets = self._get_pockets2(table)
        except Exception:
            return [float(5 * random.randint(0, 71)) for _ in range(self.DEF_DIR_N)]

        # 1) 远离前4个袋口（取反方向）
        for pk in pockets[:4]:
            v = pk - cue_p
            if float(np.linalg.norm(v)) < 1e-6:
                continue
            phi_to = (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0
            dirs.append((phi_to + 180.0) % 360.0)

        # 2) 指向球群中心
        pts = []
        for bid, b in balls.items():
            if bid == "cue" or b.state.s == 4:
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

        # 3) 随机兜底
        while len(dirs) < self.DEF_DIR_N:
            dirs.append(float(5 * random.randint(0, 71)))

        # 去重
        uniq = []
        for d in dirs:
            if all(abs(((d - u + 180) % 360) - 180) > 10 for u in uniq):
                uniq.append(d)
        return uniq[:self.DEF_DIR_N]

    def _defense_action(self, balls, my_targets, table, last_state_snapshot, my_before_left):
        dirs = self._gen_defense_dirs(balls, table)

        best_action = None
        best_score = -1e18

        for base_phi in dirs:
            for _ in range(2):  # 每方向两次扰动
                phi = float((base_phi + random.uniform(-self.DEF_JITTER, self.DEF_JITTER)) % 360.0)
                for v0 in self.DEF_V_CHOICES:
                    for s in self.POWER_SCALES_DEF:
                        action = {"V0": float(v0 * s), "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0}
                        try:
                            ok, shot = self._simulate_action(balls, table, action)
                            if not ok:
                                continue
                            score = self._eval_after_sim(shot, last_state_snapshot, my_targets, my_before_left)
                            if score > best_score:
                                best_score = score
                                best_action = action
                        except Exception:
                            continue

        if best_action is not None:
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

        # 1) 生成候选：straight（来自utils） + cut（新增）
        try:
            straight_poss = get_straight_in_all(balls, my_targets, table)
            # straight 候选加个类型标记，方便调试
            for p in straight_poss:
                p.setdefault("type", "straight")
        except Exception:
            straight_poss = []

        cut_poss = self._gen_cut_candidates(balls, my_targets, table)

        all_poss = list(straight_poss) + list(cut_poss)
        if not all_poss:
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        # 2) 结构化扩展：同一候选三档力度（不改方向，只改V0）
        expanded = []
        for p in all_poss:
            act = p.get("action", None)
            if not act:
                continue
            base_v0 = float(act.get("V0", 1.2))
            for s in self.POWER_SCALES_ATTACK:
                q = dict(p)
                q_act = dict(act)
                q_act["V0"] = float(np.clip(base_v0 * s, 0.6, 3.2))
                q["action"] = q_act
                # 稍微调整 difficulty（力度偏离越大越“难”一点点，避免乱选极端力度）
                q["difficulty"] = float(p.get("difficulty", 1e9)) + 3.5 * abs(s - 1.0)
                expanded.append(q)

        # 3) 快评估预筛，保留前 K 个进入物理仿真
        scored = [(self._fast_rank(balls, my_targets, table, p), p) for p in expanded]
        scored.sort(key=lambda x: x[0], reverse=True)
        pre = [p for _, p in scored[:self.ATTACK_PREFILTER_K]]

        # 4) 物理仿真精评，只仿真前 SIM_K 个
        best_action = None
        best_score = -1e18

        for p in pre[:self.ATTACK_SIM_K]:
            action = p["action"]
            try:
                ok, shot = self._simulate_action(balls, table, action)
                if not ok:
                    continue
                score = self._eval_after_sim(shot, last_state_snapshot, my_targets, my_before_left)
                if score > best_score:
                    best_score = score
                    best_action = action
            except Exception:
                continue

        if best_action is not None and best_score >= self.ATTACK_MIN_ACCEPT:
            return best_action

        # 5) 进攻不够好 -> 防守
        return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)
