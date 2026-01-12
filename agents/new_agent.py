import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
from .utils import get_straight_in_all
import copy
import os
import random
import signal

from .agent import Agent


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass


def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")


def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（对齐台球规则）
    """
    new_pocketed = [bid for bid, b in shot.balls.items()
                    if b.state.s == 4 and last_state[bid].state.s != 4]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed
                      if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8',
                      '9', '10', '11', '12', '13', '14', '15'}

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
    Fast+Robust NewAgent (强化版)：
    - 候选：straight + cut(ghost-ball) + 多力度
    - 快筛：cheap rank（加入 margin / deny proxy / 风险门控）
    - 精评：少量仿真
    - 鲁棒：只在不确定时触发；并带 early-stop
    """

    def __init__(self):
        super().__init__()

        # ---- 性能/搜索规模 ----
        self.SIM_TIMEOUT = 3                 # 快版：避免长滚动拖死
        self.ATTACK_PREFILTER_K = 18
        self.ATTACK_SIM_K = 7
        self.CUT_MAX_PER_TARGET = 1          # 快版：限制 cut 候选爆炸

        # ---- 动作力度档 ----
        self.POWER_SCALES_ATTACK = [0.85, 1.0, 1.18]
        self.POWER_SCALES_DEF = [0.85, 1.1, 1.35]

        # ---- 几何近似参数 ----
        self.BALL_R = 0.028575
        self.CLEARANCE = 0.002
        self.CUE_NEAR_POCKET_DIST = 0.10

        # ---- 价值函数权重 ----
        self.W_RULE = 1.0
        self.W_MY_TOP3 = 0.55
        self.W_OPP_TOP3 = 0.95
        self.W_CUE_COVER = 0.85
        self.W_CLEAR_PROGRESS = 18.0
        self.W_CUE_NEAR_POCKET = 25.0
        self.W_SCRATCH_SOFT = 25.0

        self.EASY_REF_MY = 90.0
        self.EASY_REF_OPP = 70.0
        self.TOP3_ALPHA = 0.55

        # ---- 剪枝阈值 ----
        self.ATTACK_MIN_ACCEPT = 25.0
        self.LOOKAHEAD_BASE_TH = 25.0        # 快版：减少昂贵 get_straight_in_all 触发

        # ---- 防守候选 ----
        self.DEF_DIR_N = 8
        self.DEF_JITTER = 7.0
        self.DEF_V_CHOICES = [0.9, 1.4, 1.9]

        # ---- 鲁棒评估（快版）----
        self.ROBUST_SAMPLES_ATTACK = 3
        self.ROBUST_SAMPLES_DEF = 2
        self.ROBUST_ON_TOP_M = 2
        self.ROBUST_LAMBDA_STD = 0.6

        # 差距足够大就不做鲁棒评估（更快）
        self.ROBUST_TRIGGER_GAP = 12.0

        # early-stop 容忍边界（越大越容易早停 -> 更快）
        self.ROBUST_EARLYSTOP_MARGIN = 5.0

        self.ACTION_NOISE_STD = {
            "V0": 0.08,
            "phi": 0.12,
            "theta": 0.08,
            "a": 0.003,
            "b": 0.003,
        }

        # ================== 新增：fast-rank / deny / safety 门控参数 ==================
        # fast-rank：margin 权重（裕量越大越稳）
        self.FAST_W_MARGIN = 22.0
        # fast-rank：切角惩罚（度）
        self.FAST_W_CUTANG = 0.45
        # fast-rank：力度惩罚（避免无意义大力开球）
        self.FAST_W_POWER = 6.0
        # deny proxy：对手 easy count 的权重（乘以开球风险）
        self.FAST_W_DENY = 9.0
        # deny proxy：对手 easy 统计时选多少颗最危险球
        self.DENY_K_OPP_BALLS = 4

        # 安全门控：当 cheap 层面判断“风险高且成功代理低”，转防守
        self.SAFETY_ENABLE = True
        self.SAFETY_RISK_TH = 2.0
        self.SAFETY_POT_TH = 38.0   # 0~120 量级（越大越要求稳）
        # ==========================================================================

    # ------------------ 位置/袋口工具 ------------------
    def _pos2(self, ball):
        st = getattr(ball, "state", ball)

        # pooltool canonical: state.rvw[0] -> (x,y,z)
        rvw = getattr(st, "rvw", None)
        if rvw is not None:
            try:
                r0 = rvw[0]
                if hasattr(r0, "__len__") and len(r0) >= 2:
                    return np.array([float(r0[0]), float(r0[1])], dtype=np.float64)
            except Exception:
                pass

        # fallback aliases
        for key in ("r", "xyz", "pos", "position", "p"):
            r = getattr(st, key, None)
            if r is not None and hasattr(r, "__len__") and len(r) >= 2:
                return np.array([float(r[0]), float(r[1])], dtype=np.float64)

        x = getattr(st, "x", None)
        y = getattr(st, "y", None)
        if x is not None and y is not None:
            return np.array([float(x), float(y)], dtype=np.float64)

        raise AttributeError(
            "Cannot find ball position fields. Tried state.rvw[0], state.r/xyz/pos/position/p, and state.x/y."
        )

    def _pockets_2d(self, table):
        pockets = []
        specs = getattr(table, "specs", None)
        if specs is not None and hasattr(specs, "pockets"):
            for p in specs.pockets:
                try:
                    pockets.append(np.array([float(p.x), float(p.y)], dtype=np.float64))
                except Exception:
                    pass
        if pockets:
            return pockets

        # fallback for unusual table objects
        if hasattr(table, "pockets"):
            try:
                for p in table.pockets:
                    pockets.append(np.array([float(p.x), float(p.y)], dtype=np.float64))
            except Exception:
                pass
        if pockets:
            return pockets

        # very last fallback: rough 6 pockets
        W = 2.84
        H = 1.42
        return [np.array([0.0, 0.0]), np.array([W / 2, 0.0]), np.array([W, 0.0]),
                np.array([0.0, H]), np.array([W / 2, H]), np.array([W, H])]

    # ------------------ 几何遮挡检查 ------------------
    def _dist_point_to_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 <= 1e-12:
            return float(np.linalg.norm(ap))
        t = float(np.dot(ap, ab) / ab2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _segment_clear(self, balls, a, b, exclude_ids=set()):
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:
                continue
            c = self._pos2(ball)
            d = self._dist_point_to_segment(c, a, b)
            if d <= (2 * self.BALL_R + self.CLEARANCE):
                return False
        return True

    def _segment_margin(self, balls, a, b, exclude_ids=set()):
        """
        计算线段 a->b 相对“碰撞半径(2R+clearance)”的最小裕量：
            margin = min_dist_to_segment - (2R+clearance)
        若线段上无球障碍，返回较大值。
        """
        min_d = 1e9
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:
                continue
            c = self._pos2(ball)
            d = self._dist_point_to_segment(c, a, b)
            if d < min_d:
                min_d = d
        # 这里允许 min_d=1e9 表示“没找到球”，给一个大裕量
        if min_d > 1e8:
            return 0.20  # 20cm 级别的“足够大”
        return float(min_d - (2 * self.BALL_R + self.CLEARANCE))

    # ------------------ ghost-ball cut 候选 ------------------
    def _ghost_point(self, target_pos, pocket_pos):
        v = pocket_pos - target_pos
        n = np.linalg.norm(v)
        if n < 1e-9:
            return None
        u = v / n
        return target_pos - (2.0 * self.BALL_R) * u

    def _gen_cut_candidates(self, balls, my_targets, table):
        pockets = self._pockets_2d(table)
        cue_pos = self._pos2(balls["cue"])
        cands = []

        for tid in my_targets:
            if tid not in balls:
                continue
            tb = balls[tid]
            if tb.state.s == 4:
                continue
            tpos = self._pos2(tb)

            per_target = []
            for pk in pockets:
                g = self._ghost_point(tpos, pk)
                if g is None:
                    continue
                if not self._segment_clear(balls, cue_pos, g, exclude_ids={"cue", tid}):
                    continue
                if not self._segment_clear(balls, tpos, pk, exclude_ids={tid}):
                    continue

                dv = g - cue_pos
                dn = np.linalg.norm(dv)
                if dn < 1e-9:
                    continue
                u = dv / dn
                phi = (math.degrees(math.atan2(u[1], u[0])) + 360.0) % 360.0

                v1 = (tpos - cue_pos)
                v2 = (pk - tpos)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cosang = float(np.dot(v1, v2) / (n1 * n2))
                cosang = max(-1.0, min(1.0, cosang))
                ang = math.degrees(math.acos(cosang))

                # difficulty 仍保留（用于粗排序基底）
                difficulty = 0.8 * dn + 0.015 * n2 + 0.7 * ang

                # 新增：两段裕量（更抗噪）
                m1 = self._segment_margin(balls, cue_pos, g, exclude_ids={"cue", tid})
                m2 = self._segment_margin(balls, tpos, pk, exclude_ids={tid})
                margin = float(min(m1, m2))

                action = {"V0": 1.4, "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0}
                per_target.append({
                    "type": "cut",
                    "target": tid,
                    "pocket": pk,
                    "ghost": g,
                    "difficulty": float(difficulty),
                    "cut_angle": float(ang),
                    "margin": margin,
                    "action": action
                })

            per_target.sort(key=lambda x: x["difficulty"])
            cands.extend(per_target[:self.CUT_MAX_PER_TARGET])

        return cands

    # ------------------ 机会估计（Top-3 + cover） ------------------
    def _count_remaining(self, balls, targets):
        cnt = 0
        for tid in targets:
            if tid in balls and balls[tid].state.s != 4:
                cnt += 1
        return cnt

    def _top3_easy_score(self, poss, ref):
        ds = []
        for p in poss:
            d = float(p.get("difficulty", 1e9))
            if np.isfinite(d):
                ds.append(d)
        ds.sort()
        if not ds:
            return 0.0
        e = 0.0
        a = self.TOP3_ALPHA
        for i in range(min(3, len(ds))):
            di = ds[i]
            ei = max(0.0, (ref - di))
            e += (a ** i) * ei
        return float(e)

    def _cue_cover_score(self, balls, my_targets, table):
        try:
            poss = get_straight_in_all(balls, my_targets, table)
        except Exception:
            return 0.0
        return self._top3_easy_score(poss, ref=self.EASY_REF_MY)

    # ------------------ deny proxy：对手 easy 计数（纯几何） ------------------
    def _opp_easy_count(self, balls, my_targets, table):
        """
        选择对手“最危险的 K 颗球”（按离任一袋口最近距离排序），
        若该球到某袋口无遮挡则记为 easy。
        """
        pockets = self._pockets_2d(table)

        opp = []
        for bid, b in balls.items():
            if bid in ["cue", "8"]:
                continue
            if b.state.s == 4:
                continue
            if bid in my_targets:
                continue
            pos = self._pos2(b)
            dmin = 1e9
            for pk in pockets:
                d = float(np.linalg.norm(pos - pk))
                if d < dmin:
                    dmin = d
            opp.append((dmin, bid))

        if not opp:
            return 0

        opp.sort(key=lambda x: x[0])
        top = opp[:self.DENY_K_OPP_BALLS]

        easy = 0
        for _, bid in top:
            bpos = self._pos2(balls[bid])
            # 任一袋口无遮挡就算 easy
            ok = False
            for pk in pockets:
                if self._segment_clear(balls, bpos, pk, exclude_ids={bid}):
                    ok = True
                    break
            if ok:
                easy += 1
        return int(easy)

    # ------------------ 风险评估 ------------------
    def _cue_near_pocket_penalty(self, cue_pos, table):
        pockets = self._pockets_2d(table)
        dmin = 1e9
        for pk in pockets:
            d = float(np.linalg.norm(cue_pos - pk))
            dmin = min(dmin, d)
        if dmin < self.CUE_NEAR_POCKET_DIST:
            return (self.CUE_NEAR_POCKET_DIST - dmin) / self.CUE_NEAR_POCKET_DIST
        return 0.0

    def _scratch_soft_penalty(self, cue_pos, table):
        pockets = self._pockets_2d(table)
        dmin = 1e9
        for pk in pockets:
            d = float(np.linalg.norm(cue_pos - pk))
            dmin = min(dmin, d)
        if dmin < (self.CUE_NEAR_POCKET_DIST * 1.8):
            return max(0.0, (self.CUE_NEAR_POCKET_DIST * 1.8 - dmin) / (self.CUE_NEAR_POCKET_DIST * 1.8))
        return 0.0

    # ------------------ 仿真与评估 ------------------
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

    def _noisy_action(self, action):
        a = dict(action)
        a["V0"] = float(a.get("V0", 1.2) + np.random.normal(0, self.ACTION_NOISE_STD["V0"]))
        a["phi"] = float(a.get("phi", 0.0) + np.random.normal(0, self.ACTION_NOISE_STD["phi"]))
        a["theta"] = float(a.get("theta", 0.0) + np.random.normal(0, self.ACTION_NOISE_STD["theta"]))
        a["a"] = float(a.get("a", 0.0) + np.random.normal(0, self.ACTION_NOISE_STD["a"]))
        a["b"] = float(a.get("b", 0.0) + np.random.normal(0, self.ACTION_NOISE_STD["b"]))

        a["V0"] = float(np.clip(a["V0"], 0.5, 8.0))
        a["phi"] = float(a["phi"] % 360.0)
        a["theta"] = float(np.clip(a["theta"], 0.0, 90.0))
        a["a"] = float(np.clip(a["a"], -0.5, 0.5))
        a["b"] = float(np.clip(a["b"], -0.5, 0.5))
        return a

    def _robust_eval_action(self, balls, table, action, last_state_snapshot, my_targets, my_before_left,
                           n_samples, current_best=None):
        vals = []
        for k in range(int(n_samples)):
            a = self._noisy_action(action)
            try:
                ok, shot = self._simulate_action(balls, table, a)
                if not ok:
                    vals.append(-500.0)
                else:
                    v = self._eval_after_sim(shot, last_state_snapshot, my_targets, my_before_left)
                    vals.append(float(v))
            except Exception:
                vals.append(-500.0)

            # early-stop：两次采样后开始
            if current_best is not None and k >= 1:
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                score = mu - float(self.ROBUST_LAMBDA_STD) * sd
                if score + self.ROBUST_EARLYSTOP_MARGIN < current_best:
                    break

        if not vals:
            return -500.0
        vals = np.asarray(vals, dtype=np.float32)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        return mu - float(self.ROBUST_LAMBDA_STD) * sd

    # ------------------ 新增：cheap 层成功代理 / 风险代理（不仿真） ------------------
    def _pot_proxy(self, cand):
        """
        成功代理：越大越好（0~120 量级）
        """
        d = float(cand.get("difficulty", 1e9))
        base = max(0.0, (120.0 - d))
        margin = float(cand.get("margin", 0.0))
        cut_ang = float(cand.get("cut_angle", 0.0))
        # margin 抬分，切角略减分
        return float(base + self.FAST_W_MARGIN * max(0.0, margin) - 0.25 * cut_ang)

    def _open_risk_proxy(self, cand):
        """
        开球/送分风险代理：越大越危险
        - 大力度更容易散球/摔袋
        - 大切角更不稳
        - margin 小更不稳
        """
        act = cand.get("action", {})
        v0 = float(act.get("V0", 1.4))
        cut_ang = float(cand.get("cut_angle", 0.0))
        margin = float(cand.get("margin", 0.0))

        r = 0.0
        # 力度过大惩罚
        r += max(0.0, v0 - 1.7) * 1.2
        # 切角过大惩罚
        r += max(0.0, (cut_ang - 35.0) / 25.0)
        # 裕量过小惩罚
        r += max(0.0, (0.015 - margin) / 0.015)
        return float(r)

    # ------------------ 快筛：更强 fast-rank（A+B） ------------------
    def _fast_rank(self, balls, my_targets, table, cand, opp_easy_count):
        d = float(cand.get("difficulty", 1e9))
        base = max(0.0, (120.0 - d))

        margin = float(cand.get("margin", 0.0))
        cut_ang = float(cand.get("cut_angle", 0.0))
        act = cand.get("action", {})
        v0 = float(act.get("V0", 1.4))

        # 仍保留 cover bias（昂贵，但只对少量候选触发；你若嫌慢可改成每回合算一次缓存）
        cover_bias = 0.02 * self._cue_cover_score(balls, my_targets, table)

        # deny proxy：对手 easy 越多，且你这一杆越“开球/不稳”，越要惩罚
        open_risk = self._open_risk_proxy(cand)
        deny_pen = self.FAST_W_DENY * float(opp_easy_count) * open_risk

        score = (base
                 + self.FAST_W_MARGIN * max(0.0, margin)
                 - self.FAST_W_CUTANG * cut_ang
                 - self.FAST_W_POWER * max(0.0, v0 - 1.9)
                 - deny_pen
                 + cover_bias)
        return float(score)

    # ------------------ 仿真后评估（不改） ------------------
    def _eval_after_sim(self, shot, last_state_snapshot, my_targets, my_before_left):
        rule_score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)

        balls_after = shot.balls
        my_after_left = self._count_remaining(balls_after, my_targets)
        cleared = float(my_before_left - my_after_left)
        clear_progress = cleared

        cue_pos = self._pos2(balls_after["cue"])
        near_pocket = self._cue_near_pocket_penalty(cue_pos, shot.table)
        scratch_soft = self._scratch_soft_penalty(cue_pos, shot.table)

        base_val = (self.W_RULE * rule_score
                    + self.W_CLEAR_PROGRESS * clear_progress
                    - self.W_CUE_NEAR_POCKET * near_pocket
                    - self.W_SCRATCH_SOFT * scratch_soft)

        # 剪枝：base 低就别做昂贵 lookahead
        if base_val < self.LOOKAHEAD_BASE_TH:
            return float(base_val)

        # 我方 top3
        try:
            my_poss = get_straight_in_all(balls_after, my_targets, shot.table)
        except Exception:
            my_poss = []
        my_top3 = self._top3_easy_score(my_poss, ref=self.EASY_REF_MY)

        # 对方目标集合（粗估）
        opp_targets = []
        for bid, b in balls_after.items():
            if bid in ["cue", "8"]:
                continue
            if b.state.s == 4:
                continue
            if bid not in my_targets:
                opp_targets.append(bid)

        try:
            opp_poss = get_straight_in_all(balls_after, opp_targets, shot.table) if opp_targets else []
        except Exception:
            opp_poss = []
        opp_top3 = self._top3_easy_score(opp_poss, ref=self.EASY_REF_OPP)

        cue_cover = self._cue_cover_score(balls_after, my_targets, shot.table)

        total = (base_val
                 + self.W_MY_TOP3 * my_top3
                 - self.W_OPP_TOP3 * opp_top3
                 + self.W_CUE_COVER * cue_cover)
        return float(total)

    # ------------------ 防守方向生成（不改） ------------------
    def _gen_defense_dirs(self, balls, table):
        cue_pos = self._pos2(balls["cue"])
        pockets = self._pockets_2d(table)

        dmin = 1e9
        nearest = pockets[0]
        for pk in pockets:
            d = float(np.linalg.norm(cue_pos - pk))
            if d < dmin:
                dmin = d
                nearest = pk
        away = cue_pos - nearest
        if np.linalg.norm(away) < 1e-9:
            away = np.array([1.0, 0.0])
        away_phi = (math.degrees(math.atan2(away[1], away[0])) + 360.0) % 360.0

        pts = []
        for bid, b in balls.items():
            if bid == "cue":
                continue
            if b.state.s == 4:
                continue
            pts.append(self._pos2(b))
        if pts:
            cen = np.mean(np.stack(pts, axis=0), axis=0)
            to_cen = cen - cue_pos
            if np.linalg.norm(to_cen) < 1e-9:
                to_cen = np.array([0.0, 1.0])
            cen_phi = (math.degrees(math.atan2(to_cen[1], to_cen[0])) + 360.0) % 360.0
        else:
            cen_phi = (away_phi + 90.0) % 360.0

        dirs = [away_phi, cen_phi]

        for k in range(self.DEF_DIR_N - len(dirs)):
            dirs.append(float((k * 360.0 / self.DEF_DIR_N) % 360.0))

        for _ in range(2):
            dirs.append(float(random.uniform(0, 360)))
        return dirs

    def _defense_action(self, balls, my_targets, table, last_state_snapshot, my_before_left):
        dirs = self._gen_defense_dirs(balls, table)
        best_action = None
        best_score = -1e18

        for base_phi in dirs:
            for _ in range(2):
                phi = float((base_phi + random.uniform(-self.DEF_JITTER, self.DEF_JITTER)) % 360.0)
                for v0 in self.DEF_V_CHOICES:
                    for s in self.POWER_SCALES_DEF:
                        action = {"V0": float(v0 * s), "phi": phi, "theta": 5.0, "a": 0.0, "b": 0.0}
                        try:
                            ok, shot = self._simulate_action(balls, table, action)
                            if not ok:
                                continue
                            score = self._robust_eval_action(
                                balls, table, action,
                                last_state_snapshot, my_targets, my_before_left,
                                n_samples=self.ROBUST_SAMPLES_DEF,
                                current_best=best_score
                            )
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

        remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_own) == 0:
            my_targets = ["8"]

        my_before_left = self._count_remaining(balls, my_targets)

        # 当前对手 easy count（cheap deny proxy，只算一次，供 fast-rank 使用）
        opp_easy_now = self._opp_easy_count(balls, my_targets, table)

        # 1) 候选：straight + cut
        try:
            straight_poss = get_straight_in_all(balls, my_targets, table)
            for p in straight_poss:
                p.setdefault("type", "straight")
                # 尝试给 straight 候选补充 margin（如果能定位 target）
                tid = p.get("target", None) or p.get("ball_id", None) or p.get("obj_id", None)
                if tid is not None and tid in balls and balls[tid].state.s != 4:
                    cue_pos = self._pos2(balls["cue"])
                    tpos = self._pos2(balls[tid])
                    m = self._segment_margin(balls, cue_pos, tpos, exclude_ids={"cue", tid})
                    p["margin"] = float(m)
                    p["target"] = tid
                else:
                    p.setdefault("margin", 0.0)
                p.setdefault("cut_angle", 0.0)
        except Exception:
            straight_poss = []

        cut_poss = self._gen_cut_candidates(balls, my_targets, table)

        all_poss = list(straight_poss) + list(cut_poss)
        if not all_poss:
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        # 2) 扩展力度（保留 margin/cut_angle 等信息）
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
                q["difficulty"] = float(p.get("difficulty", 1e9)) + 3.5 * abs(s - 1.0)
                expanded.append(q)

        # 3) 更强快筛（A+B）
        scored = [(self._fast_rank(balls, my_targets, table, p, opp_easy_now), p) for p in expanded]
        scored.sort(key=lambda x: x[0], reverse=True)
        pre = [p for _, p in scored[:self.ATTACK_PREFILTER_K]]

        # ================== 新增：安全门控（cheap 决策，不增加仿真） ==================
        if self.SAFETY_ENABLE and pre:
            best_cand = pre[0]
            pot = self._pot_proxy(best_cand)
            risk = self._open_risk_proxy(best_cand)

            # 若对手 easy 多，风险阈值更严格（更不该乱开球）
            risk_th = self.SAFETY_RISK_TH + 0.35 * float(opp_easy_now)

            if (risk >= risk_th) and (pot <= self.SAFETY_POT_TH):
                # 高风险、低成功代理 -> 直接防守（这是对 Pro 最有效的“少送分”策略之一）
                return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)
        # ===========================================================================

        # 4) 单次仿真粗排
        rough = []
        for p in pre[:self.ATTACK_SIM_K]:
            action = p["action"]
            try:
                ok, shot = self._simulate_action(balls, table, action)
                if not ok:
                    continue
                s1 = self._eval_after_sim(shot, last_state_snapshot, my_targets, my_before_left)
                rough.append((float(s1), action))
            except Exception:
                continue

        rough.sort(key=lambda x: x[0], reverse=True)
        if not rough:
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        # 5) 条件触发鲁棒：
        if len(rough) == 1 or (rough[0][0] - rough[1][0] >= self.ROBUST_TRIGGER_GAP):
            best_action = rough[0][1]
            best_score = rough[0][0]
        else:
            best_action = None
            best_score = -1e18
            for s1, action in rough[:self.ROBUST_ON_TOP_M]:
                s_robust = self._robust_eval_action(
                    balls, table, action,
                    last_state_snapshot, my_targets, my_before_left,
                    n_samples=self.ROBUST_SAMPLES_ATTACK,
                    current_best=best_score
                )
                if s_robust > best_score:
                    best_score = s_robust
                    best_action = action

        if best_action is None:
            return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)

        if best_score >= self.ATTACK_MIN_ACCEPT:
            return best_action

        return self._defense_action(balls, my_targets, table, last_state_snapshot, my_before_left)
