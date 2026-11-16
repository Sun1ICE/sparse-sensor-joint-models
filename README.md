# Aero-engine response reconstruction from sparse sensor locations and joint prediction of unbalance

[English](#english) · [中文](#中文)

## English

The paper uses rotational speed and triaxial complex once-per-revolution (1X) acceleration at two sensor locations to estimate rotor unbalance magnitude *U* and reconstruct vibration at the other 12 locations. Seven locations are on the turbine casing and seven on the intermediate casing. The model returns a field for all 14 locations. The paper evaluates reconstruction error at the 12 unobserved locations.

### Three model architectures

All three networks use common-reference phase normalization (Sections 3.1–3.2). The phase of the strongest observed channel sets the reference. After rotation and casing–direction RMS scaling, each node has 15 features: six real and imaginary components, three magnitudes, an observation mask, the sine and cosine of the circumferential angle, two casing indicators and normalized speed. At unobserved locations, the response features are zero while geometry and speed remain available. The model restores the original phase reference to the predicted complex field.

| Model and `family` | How the two observations inform the prediction | Trainable parameters |
| --- | --- | ---: |
| Edge-enhanced graph Transformer (`gnn`, `JointGNN`) | Three four-head attention blocks propagate 128-dimensional node states over 182 directed edges connecting distinct locations. Seven edge features encode relative angle and casing identity and enter both attention scores and message values. A node head predicts each triaxial response. A second head predicts *U* from the mean of all 14 final states. | 435,739 |
| DeepSets (`deepsets`, `JointDeepSets`) | A shared encoder maps nodes to 256 dimensions. The mean of the two observed-node representations feeds the unbalance head. The response decoder also receives each target node’s geometry and speed. | 435,975 |
| Multilayer perceptron (`mlp`, `JointMLP`) | The 14 nodes are flattened in a fixed order. The network has widths 210 → 400 → 400 → 400 → 85. Its outputs encode 84 real-valued response components and one unbalance estimate. | 439,285 |

### Four GNN ablations

Appendix B, Table B1 compares the following variants with the full joint GNN on the 21 unseen layouts of Split0.

| Paper label | `family` | Change to the full GNN |
| --- | --- | --- |
| No edge features | `gnn_no_edge` | Removes edge features from attention scores and message values. Graph connectivity, node geometry, learned attention and both prediction heads are retained. |
| Uniform aggregation | `gnn_uniform` | Assigns each of the 13 neighbors weight 1/13. Edge features still contribute to message values, and both heads remain active. |
| Response only | `gnn_response_only` | Uses the node response head alone. `unbalance_g_cm` is `None`. |
| U only | `gnn_u_only` | Uses the graph-level unbalance head alone. `response_hat_g` is `None`. |

### Inputs and outputs

Construct a model with `build_model(family, scale_g)`, then call `model(response_g, rpm, observed_mask)`. Here `B` is batch size. Use `node_spec.NODE_IDS` for node order. In this Python implementation the casing axis is **turbine first, intermediate second** (`0=T`, `1=I`). Section 3.2 of the paper numbers the two casings in the opposite order. Follow the code order when constructing tensors.

| Input | Shape | Meaning |
| --- | --- | --- |
| `scale_g` (at construction) | `[2, 3]` | Six finite, positive RMS scales in g. Rows follow code casing order. Columns are radial, tangential and axial. A trained model uses the scales fitted on its training set. |
| `response_g` | `[B, 14, 3, 2]` | Peak complex 1X accelerations in g. The last two axes are direction (radial, tangential, axial) and component (real, imaginary). Observed entries must be finite. Unobserved entries are masked before feature construction. |
| `rpm` | `[B]` | Speed in r/min, 3005–5995 inclusive. The paper’s overall PCHIP interpolation range is 3000–6000 r/min. |
| `observed_mask` | `[B, 14]` | Boolean with exactly two `True` entries per sample for the standard forward call. Each selected node supplies all three directions. |

| Output dictionary key | Shape | Meaning |
| --- | --- | --- |
| `response_hat_g` | `[B, 14, 3, 2]` or `None` | Predicted triaxial complex accelerations at all locations, in g and in the input phase frame. |
| `unbalance_g_cm` | `[B]` or `None` | Predicted unbalance magnitude in g·cm. A sigmoid scales the estimate to the model’s upper bound of 935.77 g·cm. |
| `prediction_valid`, `reference_magnitude_g` | `[B]` each | A Boolean indicating a nonzero phase reference, and the reference-channel magnitude in g. Skip predictions where the flag is false. |
| `quality_flags` | Two Boolean `[B]` tensors | `zero_observation` flags all-zero input at the observed sites. `low_signal` flags nonzero input whose largest normalized magnitude is below `1e-6`. |

### Minimal example with simulated data

Run this block from the repository root with PyTorch installed. It generates observations for I135 and I315 on the intermediate casing at 4500 r/min. Randomly initialized parameters and unit RMS scales are sufficient to check the interface. The example prints validity and shapes. To also check a common phase rotation and training-table inversion on simulated inputs, run `python -m examples.simulated_pair` after installing `requirements.txt`.

```python
import torch
from models import build_model
from node_spec import NODE_IDS

torch.manual_seed(42)
model = build_model("gnn", scale_g=torch.ones(2, 3)).eval()
pair = [NODE_IDS.index(name) for name in ("I135", "I315")]

response_g = torch.zeros(1, 14, 3, 2)
response_g[0, pair] = 0.1 * torch.randn(2, 3, 2)
observed_mask = torch.zeros(1, 14, dtype=torch.bool)
observed_mask[0, pair] = True
rpm = torch.tensor([4500.0])

with torch.inference_mode():
    out = model(response_g, rpm, observed_mask)

print(out["prediction_valid"].tolist())       # [True]
print(tuple(out["response_hat_g"].shape))     # (1, 14, 3, 2)
print(tuple(out["unbalance_g_cm"].shape))     # (1,)
```

### Paper results and available material

- **Synthetic test, Table 5.** On 21 unseen two-sensor layouts per split, the GNN’s complex-response RMSE was 0.008651 g and its unbalance MAE was 5.606 g·cm. Each split used 70 seen layouts for development.
- **Measured test and replay, Sections 4.4–4.5.** On the independent M6 condition, I135 + I315 gave a mean per-window response RMSE of 0.0109 g and an unbalance MAE of 9.58 g·cm. The second M5 rundown was a held-out record for time-domain replay. The first M5 rundown contributed to the PCHIP surface.
- **Table 7.** Run `python -X utf8 table7/reproduce.py` to reconstruct the 16 table rows from the archived group statistics, using only the Python standard library. The repository also retains processed M6 evaluation coefficients. [Table 7 statistics](table7/README.md) · [Processed input format](processed/README.md)
- **Training and replay.** The default branch distributes code and evaluation material. Trained neural-network weights, raw training recordings and the anchor archive that regenerates the study’s training arrays are excluded. Re-running the original model inference requires the corresponding local checkpoints and inputs. The simulated example above runs independently. [Training and model selection](docs/training.md) · [Layout splits](training/pair_splits/) · [Training-table inversion](docs/inversion.md)

## 中文

论文《稀疏测点下的航空发动机响应重构与不平衡量联合预测》以转速和两个测点的三向转频（1X）复加速度为输入，估计转子不平衡量幅值 *U*，并重构其余 12 个测点的振动响应。涡轮机匣和中介机匣各有 7 个测点。模型返回全部 14 个测点的复响应，论文在其中 12 个未观测点上评价重构误差。

### 三种模型架构

三个网络采用相同的公共参考相位归一化方法（论文第 3.1–3.2 节）。模型取幅值最大的观测通道作为相位参考，对复响应作相位旋转，并按机匣和方向的 RMS 尺度归一化。每个节点的 15 个特征包括 6 个实部和虚部分量、3 个幅值、1 个观测标记、2 个周向位置分量、2 个机匣编码分量和归一化转速。未观测点的响应特征置零，仍保留几何信息和转速。输出复响应时恢复输入的相位参考。

| 模型与 `family` | 如何利用两个观测点 | 可训练参数量 |
| --- | --- | ---: |
| 边增强图 Transformer（`gnn`，`JointGNN`） | 在 14 个节点间的 182 条无自环有向边上，以 3 个四头注意力块传递 128 维节点表示。7 维边特征编码相对周向角和机匣信息，同时参与注意力分数和消息值的计算。节点响应头逐点预测三向复响应，图级不平衡量头根据全部 14 个节点最终表示的均值预测 *U*。 | 435,739 |
| DeepSets（`deepsets`，`JointDeepSets`） | 共享编码器生成 256 维节点表示，仅对两个观测节点取均值。不平衡量头根据该均值预测 *U*，响应解码器还接收各目标节点的几何信息和转速。 | 435,975 |
| 多层感知机（`mlp`，`JointMLP`） | 按固定测点顺序展平 14 个节点的特征，网络层宽为 210 → 400 → 400 → 400 → 85，预测 84 个实数响应分量和 1 个不平衡量。 | 439,285 |

### 四种 GNN 消融变体

论文附录 B 表 B1 在 Split0 的 21 种未见布局上比较了以下四种变体与完整联合 GNN。

| 论文中的变体 | `family` | 相对完整 GNN 的改动 |
| --- | --- | --- |
| 移除边特征（No edge features） | `gnn_no_edge` | 从注意力分数和消息值中移除边特征，保留图连接、节点几何、可学习注意力和两个预测头。 |
| 均匀聚合（Uniform aggregation） | `gnn_uniform` | 对其余 13 个邻居分别赋予 1/13 的固定权重，消息值仍含边特征，两个预测头均保留。 |
| 仅预测响应（Response only） | `gnn_response_only` | 只使用节点响应头，`unbalance_g_cm` 为 `None`。 |
| 仅预测不平衡量（U only） | `gnn_u_only` | 只使用图级不平衡量头，`response_hat_g` 为 `None`。 |

### 输入与输出

使用 `build_model(family, scale_g)` 构造模型，再调用 `model(response_g, rpm, observed_mask)`。下表中 `B` 为批量大小，节点顺序见 `node_spec.NODE_IDS`。Python 实现的机匣顺序为**涡轮机匣在前、中介机匣在后**（`0=T`、`1=I`）。论文第 3.2 节的两个机匣编号顺序与代码相反，构造张量时请按代码顺序排列。

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `scale_g`（构造时传入） | `[2, 3]` | 6 个有限且为正的 RMS 尺度，单位为 g。行按代码中的机匣顺序排列，列依次为径向、切向和轴向。已训练模型使用在训练集上确定的尺度。 |
| `response_g` | `[B, 14, 3, 2]` | 三向 1X 复加速度峰值，单位为 g。最后两个轴依次表示方向（径向、切向、轴向）和分量（实部、虚部）。观测位置必须为有限值，未观测位置在构造特征前由掩码清零。 |
| `rpm` | `[B]` | 转速，单位为 r/min，接受 3005–5995，含端点。论文的 PCHIP 整体插值范围为 3000–6000 r/min。 |
| `observed_mask` | `[B, 14]` | 布尔张量。标准前向调用中，每个样本恰有两个位置为 `True`，每个选中测点提供三个方向的观测。 |

| 输出字典的键 | 形状 | 含义 |
| --- | --- | --- |
| `response_hat_g` | `[B, 14, 3, 2]` 或 `None` | 全部 14 点的三向复加速度预测，单位为 g，相位参考与输入一致。 |
| `unbalance_g_cm` | `[B]` 或 `None` | 不平衡量幅值，单位为 g·cm。输出经 sigmoid 映射，模型上界为 935.77 g·cm。 |
| `prediction_valid`、`reference_magnitude_g` | 各为 `[B]` | 分别表示参考幅值是否非零，以及参考通道的幅值（g）。有效性标记为假时，应跳过该样本的预测。 |
| `quality_flags` | 两个 `[B]` 布尔张量 | `zero_observation` 标记两测点输入全零的样本。对非零输入，最大归一化幅值低于 `1e-6` 时标记 `low_signal`。 |

### 模拟数据最小调用示例

在已安装 PyTorch 的环境中，从仓库根目录运行以下代码。示例在 4500 r/min 下为中介机匣的 I135 和 I315 两点生成模拟观测，采用随机初始化参数和单位 RMS 尺度，打印预测有效性和输出形状。安装 `requirements.txt` 中的依赖后，运行 `python -m examples.simulated_pair` 还可在模拟输入上检查公共相位旋转和训练表反演。

```python
import torch
from models import build_model
from node_spec import NODE_IDS

torch.manual_seed(42)
model = build_model("gnn", scale_g=torch.ones(2, 3)).eval()
pair = [NODE_IDS.index(name) for name in ("I135", "I315")]

response_g = torch.zeros(1, 14, 3, 2)
response_g[0, pair] = 0.1 * torch.randn(2, 3, 2)
observed_mask = torch.zeros(1, 14, dtype=torch.bool)
observed_mask[0, pair] = True
rpm = torch.tensor([4500.0])

with torch.inference_mode():
    out = model(response_g, rpm, observed_mask)

print(out["prediction_valid"].tolist())       # [True]
print(tuple(out["response_hat_g"].shape))     # (1, 14, 3, 2)
print(tuple(out["unbalance_g_cm"].shape))     # (1,)
```

### 论文结果与仓库资料

- **合成测试，表 5。** 每个划分包含 70 种用于开发的已见布局。在每个划分的 21 种未见双测点布局上，GNN 的复响应 RMSE 为 0.008651 g，不平衡量 MAE 为 5.606 g·cm。
- **实测与回放，第 4.4–4.5 节。** 在独立 M6 工况上，I135 + I315 的逐窗响应 RMSE 均值为 0.0109 g，不平衡量 MAE 为 9.58 g·cm。M5 第二次降速是留出的时域回放记录，第一次降速参与了 PCHIP 响应面构建。
- **表 7。** 从仓库根目录运行 `python -X utf8 table7/reproduce.py`，即可用 Python 标准库根据分组统计量重算全表 16 行。仓库另保留了处理后的 M6 评估用复系数。[表 7 统计口径](table7/README.md) · [预处理输入格式](processed/README.md)
- **训练与回放。** 默认分支提供代码和评估资料，排除了训练权重、训练原始采集记录及可重建论文训练数组的锚点归档。重新执行论文中的模型推理需在本地提供对应检查点和输入。上面的模拟示例可独立运行。[训练与选模说明](docs/training.md) · [布局划分](training/pair_splits/) · [训练表反演](docs/inversion.md)
