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
    强化版 NewAgent：
    - 进攻候选：straight-in + cut(ghost-ball)
    - 选择策略：仿真评估后取 eval 最大，而不是第一个>=阈值
    - eval：规则分 + 连攻机会 - 送球风险 + 清台进度 - 白球近袋风险
    - 防守：角度扫描×力度，挑最不送球的一杆
    """

    def __init__(self):
        super().__init__()

        # -------- 基本参数 --------
        self.SIM_TIMEOUT = 10

        # 进攻候选筛选
        self.TOPK_ATTACK = 14
        self.DIFFICULTY_MARGIN = 35.0

        # 局面评估权重
        self.W_MY_NEXT = 0.8
        self.W_OPP_NEXT = 1.7
        self.MY_EASY_REF = 85.0
        self.OPP_EASY_REF = 65.0

        self.W_CLEAR_PROGRESS = 18.0
        self.W_CUE_NEAR_POCKET = 35.0

        self.ATTACK_MIN_ACCEPT = 30.0

        # 防守扫描参数
        self.DEFENSE_ANGLE_STEP = 10
        self.DEFENSE_V_CHOICES = [0.7, 1.0, 1.3, 1.6, 2.0]
        self.DEFENSE_THETA = 5.0

        # 几何近似（切球/挡球）
        self.BALL_R = 0.028575     # 如果你们球尺度不同可改
        self.CLEARANCE = 0.002
        self.CUE_NEAR_POCKET_DIST = 0.10

    # ---------- 通用：取球位置 ----------
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

    # ---------- 通用：取袋口位置 ----------
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

    # ---------- 新增：切球候选（ghost-ball） ----------
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

                # target->pocket clear
                if not self._segment_clear(tpos, pk, balls, ignore_ids={tid, "cue"}, clearance=self.CLEARANCE):
                    continue
                # cue->ghost clear
                if not self._segment_clear(cue_p, ghost, balls, ignore_ids={tid, "cue"}, clearance=self.CLEARANCE):
                    continue

                aim = ghost - cue_p
                dist = float(np.linalg.norm(aim))
                if dist < 1e-6:
                    continue

                phi = (math.degrees(math.atan2(aim[1], aim[0])) + 360.0) % 360.0
                V0 = float(np.clip(1.0 + 0.7 * dist, 0.8, 3.0))

                dp = float(np.linalg.norm(pk - tpos))
                difficulty = 18.0 + 8.0 * dist + 3.0 * dp

                out.append({
                    "difficulty": difficulty,
                    "action": {"V0": V0, "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0},
                    "target": tid,
                    "pocket": pi,
                    "type": "cut"
                })
        return out

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

    def _evaluate_shot(self, shot: pt.System, last_state_snapshot: dict, my_targets: list, my_before_left: int):
        base = analyze_shot_for_reward(
            shot=shot,
            last_state=last_state_snapshot,
            player_targets=my_targets
        )

        if base <= -120:
            return base

        after_balls = shot.balls

        my_left_after = self._count_remaining(after_balls, my_targets)
        my_after_targets = ['8'] if (my_targets != ['8'] and my_left_after == 0) else my_targets

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

        progress = float(my_before_left - my_left_after)
        prog_term = self.W_CLEAR_PROGRESS * progress

        cue_pen = self._cue_near_pocket_penalty(after_balls, shot.table)

        return base + my_term - opp_term + prog_term - cue_pen

    # ================== 你要的：直接修改后的 decision ==================
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            print("[NewAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_own) == 0:
            my_targets = ["8"]
            print("[NewAgent] 我的目标球已全部清空，自动切换目标为：8号球")

        my_before_left = self._count_remaining(balls, my_targets)

        # 进攻候选：straight + cut
        try:
            straight_poss = get_straight_in_all(balls, my_targets, table)
        except Exception:
            straight_poss = []
        cut_poss = self._gen_cut_candidates(balls, my_targets, table)

        all_poss = list(straight_poss) + list(cut_poss)
        print(f"[NewAgent] 目标球: {my_targets}, straight={len(straight_poss)}, cut={len(cut_poss)}, total={len(all_poss)}")

        if not all_poss:
            print("[NewAgent] 没有进攻候选，转防守。")
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        ranked = sorted(all_poss, key=lambda x: float(x.get("difficulty", 1e9)))
        min_d = float(ranked[0].get("difficulty", 1e9))

        filtered = [p for p in ranked if float(p.get("difficulty", 1e9)) <= (min_d + self.DIFFICULTY_MARGIN)]
        candidates = filtered[:self.TOPK_ATTACK]

        best_action = None
        best_score = -1e18
        best_meta = None

        for i, p in enumerate(candidates, 1):
            action = p["action"]
            try:
                ok, shot = self._simulate_action(balls, table, action)
                if not ok:
                    continue

                eval_score = self._evaluate_shot(shot, last_state_snapshot, my_targets, my_before_left)
                print(f"[NewAgent] cand{i:02d} type={p.get('type','straight'):>7} "
                      f"ball {p.get('target')} -> pocket {p.get('pocket')} "
                      f"d={float(p.get('difficulty',0)):.2f} eval={eval_score:.2f}")

                if eval_score > best_score:
                    best_score = eval_score
                    best_action = action
                    best_meta = p

                if best_score >= 160:
                    break

            except Exception as e:
                print(f"[NewAgent] cand{i:02d} 模拟失败，跳过。原因: {e}")
                continue

        if best_action is not None and best_score >= self.ATTACK_MIN_ACCEPT:
            print(f"[NewAgent] ✓ 选择最佳进攻 (eval={best_score:.2f}, type={best_meta.get('type','?')})")
            return best_action

        print(f"[NewAgent] 最佳进攻不足够好 (best_eval={best_score:.2f})，转防守。")
        return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

    # ---------- 防守：角度扫描×力度，选eval最大 ----------
    def _defense_action(self, balls, my_targets, table, last_state_snapshot, my_before_left):
        best_action = None
        best_score = -1e18

        for phi in range(0, 360, self.DEFENSE_ANGLE_STEP):
            phi_jitter = float(phi) + random.uniform(-2.0, 2.0)
            for V0 in self.DEFENSE_V_CHOICES:
                action = {"V0": float(V0), "phi": float(phi_jitter), "theta": float(self.DEFENSE_THETA), "a": 0.0, "b": 0.0}
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
            print(f"[NewAgent] ✓ 选择最佳防守 (eval={best_score:.2f}) V0={best_action['V0']:.2f}, phi={best_action['phi']:.1f}")
            return best_action

        print("[NewAgent] 防守失败，随机出杆。")
        return self._random_action()
