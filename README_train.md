# 训练与使用说明

- 环境准备
  - 安装依赖：`pip install -r requirements.txt`
  - 安装 PyTorch（仅训练阶段）

- 生成数据
  - `python data_gen.py --out data/bc_data.npz --games 200 --seed 42`

- 训练行为克隆
  - `python train_bc.py --data data/bc_data.npz --out models/bc_policy.pt --meta models/meta.json --epochs 30 --batch 256 --lr 1e-3`
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
