# 双测点振动响应重构与不平衡量预测

给定两个测点的三向转频（1X）复加速度和当前转速，联合模型重构两个机匣上全部 14 个测点的三向 1X 复响应，并估计不平衡量幅值 U。模型用 PyTorch 实现，统一通过 `build_model(family, scale_g)` 创建。

## 论文复现 · Reproducing Table 7

本版本提供 GNN、MLP、DeepSets 的 27 个正式检查点（三个布局划分，每个划分三个训练种子），以及确定性的训练表反演实现。图 14 和回放计时所用的单个部署 GNN 是 `checkpoints/legacy_gnn_seed43.pt`（Split0，seed 43，epoch 1830）。[检查点清单与 SHA-256](checkpoints/manifest.json)

在仓库根目录运行以下命令。第一条表格命令只依赖 Python 标准库。其余入口使用 NumPy、SciPy 和 PyTorch。

```sh
python -X utf8 table7/reproduce.py
python -m pip install -r requirements.txt
python -m examples.simulated_pair
python replay_table7.py --device cpu --output .repro/group_sums.csv
python -X utf8 table7/reproduce.py --input .repro/group_sums.csv
```

- **核对论文数字。** `table7/reproduce.py` 从归档的 360 组误差、能量和计数重建表 7 的 16 行及三个指标的均值与样本标准差。[统计口径与字段](table7/README.md)
- **从模型重新推理。** `replay_table7.py` 读取 M6 的 200 窗 1X 复系数、同窗匹配的合成特征和冻结权重，重新评估各划分的 21 种未见布局、四种输入与目标配对（SS、SM、MS、MM）。脚本在临时目录重建约 385 MB 的训练表并执行反演。有 CUDA 的机器可将 `--device cpu` 改为 `--device cuda`。反演在 CPU 上运行。浮点结果可能随硬件和数学库略有差异。[预处理数据字段与相位约定](processed/README.md) · [反演目标函数和搜索过程](docs/inversion.md)
- **运行模拟示例。** `examples.simulated_pair` 即时生成模拟校准表和两个测点的复数输入，检查反演、冻结 GNN 和公共相位旋转。论文实测结果由上面的 M6 入口复算。
- **检查或重新执行训练。** 三种神经模型的配置、掩码、学习率筛选、早停和检查点选择见训练说明。运行 `python -m training.run --demo --output .repro/training-demo` 可在小型模拟输入上走完整个流程。运行 `python prepare_tables.py --output .repro/data` 可重建研究用合成训练与验证数组；正式训练使用 `--data-dir .repro/data`。[训练与选模说明](docs/training.md) · [三份布局划分](training/pair_splits/)

实测资料以窗口级转速、已提取的复数 1X、配重和有效性标记发布。20 kHz 原始加速度、力信号和转速采集记录均未纳入仓库。

## 三种模型架构

三种模型采用相同的输入表示。模型从两个观测点的六个通道中选取幅值最大的通道作为相位参考，从各通道的相位中减去参考相位，再按机匣和方向归一化。每个节点有 15 个特征，包括三向复响应的 6 个实数分量、3 个幅值、1 个观测标记、4 个几何特征和 1 个转速特征。输出的复响应会恢复到输入的相位坐标系。

| 模型 | `family` | 如何利用两个观测点 |
| --- | --- | --- |
| JointGNN | `gnn` | 将 14 个节点编码到 128 维，在 182 条无自环有向边上进行 3 层、每层 4 头的注意力聚合。边特征描述角度差和机匣关系，同时参与注意力和消息计算。响应头逐节点预测三向复响应，不平衡量头对全部 14 个节点的最终表示取均值后预测 U。 |
| JointDeepSets | `deepsets` | 共享编码器将节点映射到 256 维，仅对两个观测节点的编码取均值。响应头将该均值与各目标节点的几何特征和转速拼接，逐点解码。不平衡量头直接从均值预测 U。 |
| JointMLP | `mlp` | 按固定测点顺序将 14 × 15 个特征展平为 210 维，经过三个等宽隐藏层，输出 84 个响应分量和 1 个不平衡量分量。隐藏宽度按完整 JointGNN 的可训练参数量自动选取。 |

## 四种 GNN 消融变体

| `family` | 相对 `gnn` 的改动 | 预测目标 |
| --- | --- | --- |
| `gnn_no_edge` | 移除注意力分数和消息中的边属性。节点几何、图连接和基于节点特征的注意力均保留。 | 三向复响应和 U |
| `gnn_uniform` | 每个节点对其余 13 个节点采用固定权重 1/13，消息仍包含边属性。 | 三向复响应和 U |
| `gnn_response_only` | 只建立逐节点响应头。 | 三向复响应，`unbalance_g_cm` 为 `None` |
| `gnn_u_only` | 只建立图级不平衡量头。 | U，`response_hat_g` 为 `None` |

## 输入与输出

构造模型时必须提供 `scale_g`，形状为 `[2, 3]`。两行依次对应涡轮机匣和中介机匣，三列依次对应径向、切向和轴向。六个值均为有限正数，单位为 g。使用已训练模型时，应传入训练时固定的 RMS 尺度。

前向调用为 `model(response_g, rpm, observed_mask)`，其中 `B` 表示批量大小。

| 输入 | 形状与类型 | 含义 |
| --- | --- | --- |
| `response_g` | `[B, 14, 3, 2]` 浮点张量 | 振动加速度，单位为 g。测点顺序由 `node_spec.NODE_IDS` 给定，方向依次为径向、切向、轴向，末维依次为实部和虚部。观测位置必须为有限值，其余位置会被掩码清零。 |
| `rpm` | `[B]` 浮点张量 | 每个样本的转速，单位为 r/min，有效范围为 3005 至 5995，含端点。 |
| `observed_mask` | `[B, 14]` 布尔张量 | 每个样本恰有两个位置为 `True`，表示该测点的三个方向均已观测。 |

前向调用返回一个字典。

| 输出键 | 形状或类型 | 含义 |
| --- | --- | --- |
| `response_hat_g` | `[B, 14, 3, 2]` 或 `None` | 全部 14 个测点的三向复加速度预测，含两个观测点。单位及轴顺序与输入相同。 |
| `unbalance_g_cm` | `[B]` 或 `None` | 每个样本的不平衡量幅值，单位为 g·cm。模型以 sigmoid 输出乘以上界 935.77 g·cm。 |
| `prediction_valid` | `[B]` 布尔张量 | 参考通道幅值大于零时为 `True`。两个测点的响应全为零时为 `False`，此时应跳过该样本的预测数值。 |
| `reference_magnitude_g` | `[B]` 浮点张量 | 用于相位对齐的参考通道幅值，单位为 g。 |
| `quality_flags` | 字典，含两个 `[B]` 布尔张量 | `zero_observation` 标记全零观测，`low_signal` 标记观测非零且最大归一化幅值低于 `1e-6` 的样本。 |

## 最小调用示例

在已安装 PyTorch 的 Python 环境中，从仓库根目录运行以下代码。示例在 4500 r/min 下为 I135 和 I315 两个测点生成模拟复响应。模型使用随机初始化参数和单位尺度，输出仅用于演示接口。

```python
import torch

from models import build_model
from node_spec import NODE_IDS

torch.manual_seed(42)
model = build_model("gnn", scale_g=torch.ones(2, 3)).eval()

# 一个样本，只有 I135 和 I315 的三向复响应已知。
sensors = [NODE_IDS.index(name) for name in ("I135", "I315")]
response_g = torch.zeros(1, 14, 3, 2)
response_g[:, sensors] = 0.1 * torch.randn(1, 2, 3, 2)
observed_mask = torch.zeros(1, 14, dtype=torch.bool)
observed_mask[:, sensors] = True
rpm = torch.tensor([4500.0])

with torch.inference_mode():
    result = model(response_g, rpm, observed_mask)

print(result["prediction_valid"].tolist())       # [True]
print(tuple(result["response_hat_g"].shape))     # (1, 14, 3, 2)
print(tuple(result["unbalance_g_cm"].shape))     # (1,)
```
