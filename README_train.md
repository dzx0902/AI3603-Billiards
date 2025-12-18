# 训练与使用说明

- 环境准备
  - 安装依赖：`pip install -r requirements.txt`
  - 安装 PyTorch（仅训练阶段）

- 生成数据
  - 推荐：`python data_gen.py --out data/bc_data.npz --games 600 --seed 42 --min_reward 10 --positive_ratio 0.6`
  - 统计查看：生成 `data/bc_stats.json` 用于口袋与犯规占比评估

- 训练行为克隆
  - 推荐：`python train_bc.py --data data/bc_data.npz --out models/bc_policy.pt --meta models/meta.json --epochs 60 --batch 256 --lr 5e-4 --wd 1e-5 --scheduler 1 --patience 8 --val_split 0.1 --dim_weights 1,1,1,0.8,0.8`
  - 可视化：训练会生成 `models/logs/metrics.json` 与 `models/logs/loss_curve.png`

- 评测
  - `python evaluate.py`
  - 加载优先级：`models/sac_policy.pt` → `models/bc_policy.pt` → 兜底
  - 日志打印策略来源

- 产物
  - `models/bc_policy.pt`：TorchScript 模型
  - `models/meta.json`：状态归一化参数与动作边界
  - `data/bc_data.npz`：专家数据集

- 备注
  - 本阶段交付 BC-only；待你完成评测并反馈后，再执行 SAC 微调并交付相应脚本与模型。
  - 快速自检：`python bc_eval_sanity.py --games 10 --noise 0 --render 0` 查看口袋/犯规统计（非正式评测）
  - 如需强制用 CPU 推理：PowerShell 执行 `setx CUDA_VISIBLE_DEVICES ""` 或运行前 `PowerShell: $env:CUDA_VISIBLE_DEVICES="" ; python evaluate.py`
