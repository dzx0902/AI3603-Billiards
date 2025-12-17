## 目标与约束
- 目标：实现“先BC闭环跑通、后SAC微调”的两阶段改造，使 `python evaluate.py` 在40局稳定运行，BC阶段即可完成评测闭环；待您验收后再加SAC提升胜率。
- 约束：只修改 `agent.py` 的 `NewAgent`，不改 `evaluate.py` 与 `poolenv.py`；可新增训练/工具脚本；`decision` 始终返回合法动作；无模型时可用。

## 当前仓库要点
- 评测流程：`evaluate.py` 将 `NewAgent` 与 `BasicAgent` 对战（evaluate.py:31-71）。
- 环境接口：`PoolEnv.reset/get_observation/take_shot/get_done`（poolenv.py:196-467）。
- 奖励函数：`analyze_shot_for_reward`（agent.py:27-115）。
- Bayes参考：动作边界 `pbounds`（agent.py:160-167）。

## 阶段一：BC-only 闭环改造（先执行）
### 修改任务列表（BC阶段）
- 实现固定维度状态编码与掩码（白球、我方/对方球、黑8、桌面与袋口）。
- 完成动作归一化/反归一化与严格裁剪逻辑。
- 在 `NewAgent` 中加载并推理 BC 模型（无模型则启发式/BasicAgent兜底）。
- 加入轻量邻域搜索微调（8~16个邻域评估，设置上限与异常保护）。
- 新增 `data_gen.py`：用 `BasicAgent`/启发式生成专家数据，含并行与种子。
- 新增 `train_bc.py`：MLP监督训练，保存 `models/bc_policy.pt` 与 `models/meta.json`。
- 新增 `utils_state.py`（状态构造/归一化）、可选 `replay_buffer.py`（后续SAC可复用）。
- 新增 `README_train.md`：一键命令与模型放置说明。

### 设计与实现要点
- 状态向量：固定顺序拼接，缺失球用0并 `mask=0`；包含 `cue.xy`、我方/对方球位置+mask、`8`号球位置+mask、`table.w/l` 与6袋口中心坐标。
- 动作空间：训练用 `[-1,1]`；推理反归一化并裁剪到 `V0[0.5,8.0]`、`phi[0,360)`（取模）、`theta[0,90]`、`a,b[-0.5,0.5]`（agent.py:160-167）。
- 轻量邻域搜索：在沙盒 `pt.System` 上评估，打分用 `analyze_shot_for_reward`（agent.py:27-115）；`self.max_eval_per_move` 控制次数，异常记低分；超时/异常回退到策略原始动作或启发式。
- 可靠性：严格裁剪、防 `NaN`、异常捕获、失败重试与兜底，确保 `decision` 不阻塞。

### 产物与验证（BC阶段）
- 产物：`models/bc_policy.pt`、`models/meta.json`、`data/*.npz`。
- 命令：
  - 生成数据：`python data_gen.py --out data/bc_data.npz --games 200 --seed 42`
  - 训练BC：`python train_bc.py --data data/bc_data.npz --out models/bc_policy.pt`
  - 评测：`python evaluate.py`（自动加载BC；日志打印策略来源）
- 验收：40局完成且不中断；在仅BC模型存在时闭环可用；`decision` 始终合法。

## 阶段二：SAC 微调（待您检测完成后再执行）
### 修改任务列表（SAC阶段）
- 编写最小 Gym-like 包装（不改 `evaluate.py`）：`reset/step` 复用 `PoolEnv` 与 `analyze_shot_for_reward`。
- 训练 `train_sac.py`：从 BC 权重初始化 actor，critic 随机；标准经验回放与目标网络。
- 保存 `models/sac_policy.pt` 并更新 `models/meta.json`；`NewAgent` 优先加载 `sac_policy.pt`。
- 保留与BC相同的裁剪/邻域微调与异常保护逻辑。

### 设计与实现要点
- `reset()`：用 `PoolEnv.reset` 生成初始局面并返回状态向量（poolenv.py:196-233）。
- `step(action)`：反归一化→裁剪→调用 `PoolEnv.take_shot`（poolenv.py:240-467），奖励用 `analyze_shot_for_reward`；返回 `(next_state, reward, done, info)`。
- 训练：CPU/GPU自适应；学习率与温度等超参可在README中给出推荐值；可设置训练回合与收敛准则。

### 产物与验证（SAC阶段）
- 产物：`models/sac_policy.pt`、更新的 `models/meta.json`。
- 命令：`python train_sac.py --init models/bc_policy.pt --out models/sac_policy.pt`
- 验收：在存在SAC模型时评测优先加载SAC；40局完成；日志显示策略来源为SAC。

## 路径与文件结构
- `agent.py`（只改 `NewAgent`）
- `data_gen.py`、`train_bc.py`、`train_sac.py`、`utils_state.py`、`replay_buffer.py`（可选）
- `models/`：`bc_policy.pt`、`sac_policy.pt`、`meta.json`
- `README_train.md`

## 验收标准
- 无模型：`NewAgent` 兜底运行40局，打印使用fallback。
- 仅BC：加载 `bc_policy.pt` 完成评测，输出使用BC策略日志。
- 有SAC：优先加载 `sac_policy.pt` 完成评测，输出使用SAC策略日志。
- 始终合法动作、无阻塞；异常捕获与回退到位。

## 风险与备选
- 仿真偶发异常/NaN：评估时记低分并回退；裁剪与取模保证安全。
- 数据质量不足：增加专家轨迹、使用启发式增强；调整 MLP容量与正则。
- 训练不收敛：调学习率/温度/批量大小；延长迭代或采用早停。

—— 按您的要求：先交付并运行“BC-only 闭环”，待您检测完成后，再进行“SAC 微调”阶段的改造与交付。