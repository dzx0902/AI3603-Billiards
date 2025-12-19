## 聚焦两阶段改造（优先交付 BC-only）

* 执行顺序：先完成并交付 BC-only 闭环，待你运行评测通过后，再进行 SAC 微调改造与交付。

## 阶段一：BC-only 闭环（立即实施）

### 代码分析与不符点

* 现状：`NewAgent` 为空实现；缺少 `data_gen.py`、`train_bc.py`、`utils_state.py`、`models/` 与 `meta.json`。

* 风险：分支已改动 `BasicAgent`（超时保护），与“只改 NewAgent”约束不符；后续将逻辑迁移到 `NewAgent`，并记录变更说明。

### 修改任务清单

* `agent.py`（只改 `NewAgent`）：

  * 加载优先级：`models/sac_policy.pt` → `models/bc_policy.pt` → 兜底（日志显示来源）。

  * 集成状态编码与动作反归一化+裁剪；轻量邻域搜索（8\~16 个近邻，异常低分，评估上限）确保鲁棒性。

  * 兜底逻辑：无模型或推理失败时使用启发式/`BasicAgent` 决策；`decision` 总返回合法动作。

* `utils_state.py`：

  * 固定维状态向量构造（白球、己方/对方球+mask、黑8+mask、桌面尺寸与袋口坐标）；`normalize_state`、`action_denorm`/`action_norm`；`meta.json` 读写。

* `data_gen.py`：

  * 用 `BasicAgent` 生成轨迹 `(state, action_norm, reward, next_state, done)`；支持种子与并行；保存 `.npz`。

* `train_bc.py`：

  * 轻量 MLP（Tanh 输出）；MSE 训练；保存 TorchScript `models/bc_policy.pt` 与 `models/meta.json`；不要求 `evaluate.py` 安装 `torch`（训练环境单独安装）。

* `README_train.md`：

  * 三步命令：生成数据→训练 BC→评测；说明模型路径与加载优先级；依赖隔离策略（训练需 `torch`）。

### 依赖与环境注意

* 训练依赖：`torch`（仅训练环境）。

* 评测依赖：沿用现有 `requirements.txt`。如遇 NumPy/ABI 问题，建议：统一为 `numpy=1.26.x` 与匹配的 `pyarrow/pandas`，并强制重装 `pooltool-billiards`。

### 验证与交付

* 交付物：`agent.py`（改 `NewAgent`）、`utils_state.py`、`data_gen.py`、`train_bc.py`、`README_train.md`、`models/`（训练后生成）。

* 验证：在仅 `bc_policy.pt` 存在时，运行 `python evaluate.py` 完成 40 局；`decision` 始终合法；日志显示“策略来源：BC”。

## 阶段二：SAC 微调（你的 BC 评测通过后执行）

### 修改任务清单

* `train_sac.py` 与可选 `replay_buffer.py`：

  * Gym-like 包装（复用 `PoolEnv.reset/step` 与 `analyze_shot_for_reward`）；SAC 训练，actor 从 BC 初始化、critic 随机；保存 `models/sac_policy.pt`。

* `agent.py`（`NewAgent` 已支持）：

  * 优先加载 `sac_policy.pt`；其他逻辑复用 BC 推理与邻域搜索。

* 更新 `README_train.md`：

  * 增加 SAC 训练命令与注意事项。

### 验证与交付

* 交付物：`train_sac.py`、`models/sac_policy.pt`（训练生成）。

* 验证：存在 `sac_policy.pt` 时 `python evaluate.py` 完成 40 局；日志显示“策略来源：SAC”。

## 质量保证

* 单元测试：状态编码维度与顺序、动作裁剪边界、反归一化正确性。

* 集成测试：`data_gen.py`→`train_bc.py`→`evaluate.py` 小样本跑通；`decision` 始终返回。

* 回归测试：确认 `evaluate.py` 不改且 40 局稳定完成；记录 `BasicAgent` 改动的影响与后续合规迁移计划。

