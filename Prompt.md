你是一个代码实现 agent。请在不修改 evaluate.py 的前提下，只允许修改 agent.py 中的 NewAgent 类，并允许新增训练与工具脚本文件，实现一个“BC-only 先闭环跑通，再 SAC 微调”的混合强化学习方案，以在 evaluate.py 的 40 局对战评测中稳定达到 >= 70% 胜率（尽可能更高）。

# 你可以使用的文件
- agent.py（已存在）：包含 BasicAgent、NewAgent（只能修改 NewAgent 类；BasicAgent 不改）
- evaluate.py（已存在）：评测脚本（绝对不能修改）
- GAME_RULES.md（已存在）：规则与奖励/犯规定义（只读）
你可以新增文件（例如 train_bc.py / train_sac.py / data_gen.py / utils_state.py 等），并在 NewAgent 内按需 import，但评测时 evaluate.py 只会 import agent.py 并调用 NewAgent.decision。

# 运行约束与目标
- 必须保证：`python evaluate.py` 可以直接运行（假设用户已安装依赖）。
- NewAgent 的 `decision(self, balls, my_targets, table)` 返回 dict: {'V0','phi','theta','a','b'}，数值合法。
- NewAgent 在没有训练模型文件时必须可用（fallback）：至少能像 BasicAgent 一样工作（或退化为启发式+局部优化）。
- BC-only 阶段：必须提供完整闭环（数据生成 -> 训练 -> 在 NewAgent 加载并推理 -> evaluate 可跑）。
- SAC 微调阶段：必须提供脚本与流程（使用 BC 初始化 policy，再继续训练），最终产出 NewAgent 可加载的模型权重。
- 不引入巨量手工步骤：脚本可一键运行（例如 `python data_gen.py ...`，`python train_bc.py ...`，`python train_sac.py ...`）。
- 不要求 GPU，但如果有 GPU 应可用（torch 自动选择 cuda）。

# 关键实现要求（必须做到）
## A. 状态表示（State Encoding）
从 pooltool 的 `balls` 和 `table` 构造一个固定维度的状态向量，至少包含：
1) 白球位置 (x,y)
2) 目标球（my_targets）中仍在桌面的球的位置 (x,y)，并用 mask 标记是否存在
3) 对手球的位置 (x,y) + mask
4) 黑8位置 (x,y) + mask
5) 可选：桌面尺寸、袋口坐标（可固定常量或从 table 中读取并拼入）
要求：对球的排列顺序固定（例如按球号排序 1..15，再 cue），缺失球用 0 填充并用 mask 表示。

## B. 动作表示（Action Encoding）
动作是连续 5 维：a = [V0, phi, theta, spin_a, spin_b]。
训练时建议把它归一化到 [-1,1] 或 [0,1]，推理时再反归一化到合法区间（与 pbounds 一致或更保守）。
必须实现动作裁剪，避免非法值导致仿真异常。

## C. BC-only 闭环
### 1) 专家数据生成（新增脚本 data_gen.py）
- 用 BasicAgent 或你在 NewAgent 内的启发式+局部优化作为“专家”生成数据。
- 生成若干局对战轨迹（至少数千 step 的 (state, action) 对），保存为 .npz 或 .pt。
- 每条样本至少包含：state、action、可选 reward/next_state/done（为后续 SAC 用）。
- 生成时允许跑 pooltool simulate（CPU），允许并行（multiprocessing），要有随机种子控制。

### 2) 行为克隆训练（新增脚本 train_bc.py）
- 用 PyTorch 实现一个轻量 MLP policy（可加 layernorm / tanh 输出）。
- 损失：MSE(action_pred, action_expert)（在归一化空间），可加 action bounds penalty。
- 训练完保存权重到 `models/bc_policy.pt`（路径可配置）。
- 同时保存一个 `models/meta.json`（包含归一化参数、状态维度、动作缩放范围等）。

### 3) 在 NewAgent 中加载 BC policy 并推理
- NewAgent.__init__ 尝试加载 `models/bc_policy.pt`；若不存在则 fallback。
- NewAgent.decision 中：
  - 构造 state -> policy 推理 -> 得到 action
  - 对 action 做裁剪
  - 可选：做 very small 的局部随机扰动搜索（例如采样 8~16 个邻域动作，用现有 reward 函数评估挑最优），作为“轻量规划微调”，提高稳定性。
  - 若仿真/评估失败次数过多则 fallback 到 BasicAgent。

## D. SAC 微调（新增脚本 train_sac.py）
目标：在 BC 基础上，用 SAC 继续提升胜率与稳定性。
要求：
- 环境：你需要写一个最小 Gym-like wrapper（不改 evaluate.py）：
  - reset(): 生成一个初始局面（可直接复用项目里提供的初始化方式；如果评测代码里有生成逻辑，尽量复用其函数/类）
  - step(action): 调用 pooltool simulate 一杆，利用 GAME_RULES/agent.py 里的 analyze_shot_for_reward 或等价逻辑得到 reward，并返回 next_state, reward, done, info
- 初始 policy/critic：从 BC policy 初始化 actor；critic 随机初始化。
- Replay buffer：可以直接用常见实现。
- 训练脚本输出：`models/sac_policy.pt`（actor 权重）+ `models/meta.json` 更新。
- NewAgent 推理优先加载 sac_policy.pt，若不存在则加载 bc_policy.pt，否则 fallback。

注意：SAC 的训练可以较慢，但脚本必须可运行，不要求你在这里真的训到 70%，但代码必须是完整闭环可执行。

# E. 可靠性与防卡死（必须处理）
- pooltool 仿真偶尔会报 warning/返回 NaN。你必须：
  - 对输入动作做严格裁剪
  - 捕获 simulate 异常并给予低 reward
  - 在 NewAgent 内设置最大仿真次数/超时保护（例如局部搜索最多评估 N 次）
- decision 必须始终返回一个合法动作，不能 hang。
- 需要在 NewAgent 中加一个 `self.max_eval_per_move` 之类的上限，保证 evaluate 不会“卡死”。

# 交付物清单
1) 修改后的 agent.py：只改 NewAgent 类，其余不动。
2) 新增脚本：
   - data_gen.py（生成 BC 数据）
   - train_bc.py（训练 BC）
   - train_sac.py（SAC 微调）
   - 可选 utils_state.py / utils_env.py / replay_buffer.py 等
3) README_train.md（新增）：写清楚三步命令：
   - (1) 生成数据
   - (2) 训练 BC
   - (3) SAC 微调
   并说明模型文件放在哪里，evaluate.py 如何自动使用（只要 NewAgent 加载到即可）。
4) NewAgent 必须做到：
   - 优先使用 `models/sac_policy.pt` -> `models/bc_policy.pt` -> fallback
   - 推理 + 裁剪 + 轻量邻域搜索（可选但推荐）
   - 绝不修改 evaluate.py

# 质量要求与验收
- `python evaluate.py` 可直接运行并完成 40 局，不崩溃、不无限等待。
- 在仅 BC 模型存在时能运行（BC-only 闭环）。
- 在 SAC 模型存在时能运行（SAC 闭环）。
- 输出日志中可打印当前使用的策略（SAC/BC/fallback）。
- 代码结构清晰，带必要注释，尽量不引入重依赖（只用 numpy、torch、pooltool 及项目已有依赖）。

现在请你根据以上要求，直接生成最终代码修改与新增文件内容（以多文件形式输出），确保可复制粘贴到项目目录中使用。
