# CUHK-X Small Model Track 技术路线 v1.1

> 文档性质：团队讨论稿与第一阶段执行指南  
> 更新时间：2026-07-14  
> 当前目标：先建立可信、可复现的跨受试者基线，再逐步扩展为小于 100 MB 的多模态模型。

## 0. 一页结论

本赛题是 40 类、多模态、跨受试者 Human Activity Recognition（HAR）。深度数据包含 Raw Depth 与 `Depth_Color` 两种表示，此外还有 `IR`、`Thermal`、`Skeleton`、`IMU`、`Radar` 等候选模态；测试样本可能缺少部分模态。主指标是 Accuracy，最终模型需要控制在 100 MB 以内。

我们不应一开始就把六种模态、Transformer、蒸馏全部堆在一起。推荐按下面的顺序推进：

```text
阶段 0：数据解压、审计、跨受试者验证划分、统一训练框架
    ↓
阶段 1A：Depth_Color + ResNet18 + mean pooling，跑通最小视觉基线闭环
    ↓
阶段 1B：比较 Raw Depth 与 Depth_Color，并比较 ImageNet 预训练与随机初始化
    ↓
阶段 1C：比较 mean/attention pooling、TCN、GRU、TSM 等时序方案
    ↓
阶段 2：Skeleton 单模态基线 + Depth/Skeleton OOF 后融合
    ↓
阶段 3：Depth/Skeleton 门控特征融合，并依次评估 IR、IMU、Thermal、Radar
    ↓
阶段 4：必要时使用小型 Transformer、对比学习和缺失模态鲁棒训练
    ↓
阶段 5：建立强教师后再进行蒸馏
    ↓
阶段 6：压缩并得到小于 100 MB 的最终可复现模型
```

### 当前候选模块池与可能的最终组合

以下结构表示研究候选池，不代表最终模型一定同时使用全部模态。最终模型很可能只保留 2～4 种具有稳定边际收益的模态。某个模态是否保留，取决于其单模态能力、与主模态的互补性、cross-subject 多折增益、最差 fold 表现、模型大小与推理成本，以及数据完整性和实现风险。

```text
主视觉模态
Raw Depth 或 Depth_Color
→ 轻量预训练 CNN
→ 通过实验选出的轻量时序模块

第一辅助模态：优先验证 Skeleton
→ 人体中心归一化 + 坐标/骨骼/速度特征
→ 轻量 TCN 或 Ta-CNN

其他候选模态：IMU、IR、Thermal、Radar
→ 分别建立单模态 baseline
→ 通过 OOF 后融合和消融判断是否保留

融合
→ 先做 logits 后融合
→ 再做门控特征融合
→ 必要时再使用小型 Transformer
→ 40 类
```

Raw Depth 与 Depth_Color 谁更适合作为主视觉输入，TSM、TCN、GRU 或其他轻量时序模块谁更适合作为主干的一部分，都必须由 subject-disjoint 多折实验决定。每一种新增模态和复杂模块也必须逐一通过单模态、OOF 后融合与消融验证，才能进入最终模型。

---

## 1. 已知条件与工作假设

### 1.1 已知赛题条件

- 任务：一个 clip 对应一个动作标签，共 40 类，标签为 `0–39`。
- 训练标签包含在路径中：`<modality>/<action>/<user>/<trial>/...`。
- 测试集共有 405 个匿名 clip。
- 深度数据提供 Raw Depth 与 `Depth_Color` 两种表示；其他主要模态包括 `IR`、`Thermal`、`Skeleton`、`IMU`、`Radar`。
- `Depth_Color` 是由深度数据进行伪彩色映射得到的三通道图像，便于直接利用 ImageNet 预训练 CNN；Raw Depth 保存更直接的距离与几何信息，可能具有更明确的物理含义。
- Raw Depth 与 `Depth_Color` 谁更有利于 cross-subject 泛化不能凭直觉决定，必须在阶段 1 做正式对照。
- 不同模态采样率不同，部分 clip 缺少某些模态。
- 主指标：Accuracy；Macro-F1、每类 Recall、混淆矩阵只作为诊断指标。
- 训练和测试受试者不同，因此真正难点是跨受试者泛化，而不是随机 clip 划分下的分类。
- 模型文件必须不超过 100 MB。

### 1.2 团队补充确认

- ImageNet 等公开预训练权重合规，可以使用。
- 三名成员至少各有一张 RTX 4070 Laptop，另有 RTX 3090；必要时可使用实验室服务器。
- Codex、Cursor 等 Agent 可以辅助开发，但不能用于生成或猜测测试集标签，也不能代替合法的数据标注流程。（再确认一下）

### 1.3 当前算力分配建议

- RTX 4070 Laptop：数据管线、单模态模型、单折快速实验、推理复现。
- RTX 3090：主模型、多模态联合训练、较大 batch、多折正式实验。
- 实验室服务器：后期多折并行、多个随机种子、教师模型和大规模消融。

---

## 2. 全程必须遵守的实验原则

### 2.1 验证划分优先于模型选择

禁止把同一个用户的 clip 同时放入训练集和验证集。否则验证分数会利用个人体型、动作习惯、衣着、视角和房间背景，产生明显虚高。

推荐建立两套验证配置：

1. `dev_fold`：固定留出 4 个用户，用于快速开发，所有人共用同一份划分。
2. `cv_folds`：4–5 个 subject-disjoint folds，用于确认重要改动和最终模型。

如果解压后确认用户 `1–9` 与 `16–24` 分别对应两个环境或两组采集域，则每个验证 fold 应从两组中分别留用户，尽量模拟线上测试的用户构成。最终划分需要结合每位用户的类别覆盖情况生成，不能机械套用普通 `KFold`。

模型比较至少记录：

- 各 fold Accuracy；
- 平均 Accuracy；
- 最差 fold Accuracy；
- Macro-F1 与每类 Recall；
- 参数量、FP32 权重大小、单 clip 推理时间；
- 各种缺失模态条件下的 Accuracy。

### 2.2 公榜只用于低频验证

只有满足以下条件的实验才值得提交：

- 本地跨用户平均分有稳定提升；
- 多个 fold 方向一致；
- 最差 fold 没有明显退化；
- 推理流程已经固定并且不会产生 CSV 格式错误。

不能用公开榜反复选择学习率、帧数、随机种子或融合权重。测试集只有 405 个 clip，榜单波动可能很大，而且决赛仍会使用全新的跨受试者数据。

### 2.3 每个阶段都要留下可复现产物

每次有效实验至少保存：

```text
config.yaml
splits.json
train.log
metrics.json
oof_predictions.csv
confusion_matrix.png
checkpoints/best.pth
model_profile.json        # 参数量、权重大小、FLOPs/延迟
```

代码中不得把绝对数据路径写死；路径、fold、随机种子、模型和超参数全部由配置文件控制。

---

## 3. 阶段 0：数据与实验地基

这一阶段看起来不像“做模型”，但它决定后面所有分数是否可信。预计用 1–3 天完成。

### 3.1 解压与完整性检查

当前训练集由以下文件组成：

```text
HAR.z01 ... HAR.z08
HAR.zip
```

九个文件必须位于同一目录，并保持标准名称。建议使用 7-Zip 直接从 `HAR.zip` 解压，避免先合并为另一个 40GB+ 的临时文件造成额外磁盘占用。测试集解压 `small_model_track_test.zip` 即可。

解压后预期结构：

```text
Training/data/HAR/data/<modality>/<action>/<user>/<trial>/<files>
Testing/data/small_model_track_test/<sample_id>/<modality>/<files>
```

解压完成后需要检查：

- 40 个动作目录是否齐全；
- 训练用户集合和测试目录数量是否正确；
- 是否存在 0 字节文件、损坏图片或无法解析的 CSV；
- 各传感器模态及 Raw Depth/`Depth_Color` 两种深度表示的目录名是否与实际数据一致；
- 一个 trial 在不同模态中的键能否稳定对应。

### 3.2 建立统一 manifest

训练时不允许每个 Dataset 临时遍历几百万个文件。首先扫描一次数据，生成 `train_manifest.parquet/csv` 和 `test_manifest.parquet/csv`。

每一行对应一个 clip，至少包含：

```text
clip_id
action_id
action_name
user_id
trial_id
raw_depth_path / raw_depth_num_frames
depth_color_path / depth_color_num_frames
ir_path / ir_num_frames
thermal_path / thermal_num_frames
skeleton_path / skeleton_length
imu_path / imu_length
radar_path / radar_length
has_raw_depth / has_depth_color / has_ir / has_thermal / has_skeleton / has_imu / has_radar
```

同时输出数据审计报告：

- 每类 clip 数量；
- 每位用户的 clip 数量与类别覆盖；
- 每种模态的可用率；
- clip 长度、帧率、分辨率分布；
- 缺失模态组合频率；
- 测试集与训练集在长度、分辨率、模态缺失方面的分布差异。

### 3.3 统一代码框架

推荐目录：

```text
configs/
src/
  data/
  models/
  losses/
  train.py
  validate.py
  infer.py
scripts/
checkpoints/
outputs/
tests/
```

所有模态 Dataset 最终返回统一结构：

```python
{
    "inputs": {"depth": ..., "skeleton": ..., ...},
    "modality_mask": ...,
    "label": ...,
    "clip_id": ...,
    "user_id": ...,
}
```

这样后续添加模态时不需要重写训练循环。

### 3.4 阶段退出条件

- 数据可完整解压并通过基础完整性检查；
- manifest 与数据审计报告生成成功；
- subject-disjoint 划分固定并共享给全队；
- 一个最小 Dataset/DataLoader 能读出 batch；
- 能用随机 logits 走通验证、推理和 submission CSV 生成流程。

---

## 4. 阶段 1：深度输入与轻量时序模型

阶段 1 分为 1A、1B、1C 三层。先用最简单的 `Depth_Color + ResNet18 + mean pooling` 跑通完整闭环，再比较 Raw Depth 与 `Depth_Color`，最后才比较 TSM、TCN、GRU 和 attention 等时序模块。这样才能把数据问题、输入表示收益和时序建模收益分开判断。

`Depth_Color` 是将深度数据进行伪彩色映射后得到的三通道图像，最容易直接接入 ImageNet 预训练 CNN。Raw Depth 则保留更直接的距离与几何信息，物理含义更明确，但它与自然图像的通道结构和数值分布差异更大。二者谁更有利于本赛题，尤其是 cross-subject 泛化，不能仅凭直觉确定。

### 4.1 阶段 1A：最小可运行视觉基线

```text
Depth_Color clip
→ 分段采样若干帧
→ ImageNet 预训练 ResNet18 逐帧提取特征
→ mean pooling
→ Linear
→ 40 类
```

这个模型的任务不是追求最高分，而是建立最简单、最容易排查错误的端到端系统。它首先验证：

- 数据读取是否正确；
- 帧的时间顺序与分段采样是否正确；
- `action_id` 与路径标签映射是否正确；
- subject-disjoint 训练/验证划分是否真正按用户隔离；
- 训练、验证、推理和 submission 流程是否全部打通。

最小基线将每帧 ResNet18 输出组织为 `[B,T,512]`，直接在时间维求平均得到 `[B,512]`，再进行 40 类分类。这里暂不加入 TSM、GRU、TCN、attention、复杂损失或多模态模块；只有简单基线稳定后，后续模块的真实收益才可测量。

输入与采样建议：

- 每个 clip 均匀划分为 8 个时间段；
- 训练时每段随机取 1 帧；
- 验证和推理时每段取中心帧；
- 分辨率先用 `224×224`；显存紧张时可先用 `160×160` 调试；
- 保留原始时间顺序，不允许把帧随机打乱；
- 不做时间反转增强，因为 `Stand_up/Sit_down/Lie_down` 等类别可能依赖动作方向。

`Depth_Color` 第一版使用 ImageNet mean/std，以匹配预训练骨干；增强仅使用适度 `RandomResizedCrop`、水平翻转、小幅平移/旋转和轻量 Random Erasing。不要使用强 hue/saturation 变化，因为伪彩色可能编码深度，而不是自然颜色。

### 4.2 阶段 1B：Depth 输入形式对照

保持帧采样、ResNet18 主干、mean pooling、训练划分和大部分超参数一致，至少比较以下三种输入：

#### 方案 1：Depth_Color 三通道输入

```text
Depth_Color
→ 原始 ImageNet 预训练 ResNet18
→ mean pooling
→ 40 类
```

这是最容易建立、最接近官方视觉基线风格的起点。优势是能完整使用 ImageNet 预训练第一层；风险是伪彩色映射可能引入与真实几何无关的颜色模式。

#### 方案 2：Raw Depth 单通道输入

```text
Raw Depth [1,H,W]
→ 将 ResNet18 第一层卷积改为单通道
→ mean pooling
→ 40 类
```

若使用 ImageNet 预训练，可将原始第一层三通道权重沿通道维做平均或等价初始化，得到单通道卷积权重；后续层继续使用预训练参数。Raw Depth 的无效值、量纲、有效距离范围和归一化方式必须先通过数据审计确定，不能直接按普通灰度图处理。

#### 方案 3：Raw Depth 复制或合理映射为三通道

```text
Raw Depth
→ 复制为三通道，或使用固定且可复现的三通道映射
→ 原始 ImageNet 预训练 ResNet18
→ mean pooling
→ 40 类
```

该方案保留原始 ImageNet 第一层结构，开发简单，但三个通道可能高度冗余。任何“合理映射”都必须只依赖 Raw Depth 本身并固定应用于训练/验证/测试，不能使用测试标签或按样本人工调整。

#### 预训练对照

以上输入形式需要把下面的实验列为重要对照：

```text
ImageNet 预训练
vs
随机初始化
```

第一轮先完成三种输入在 ImageNet 预训练条件下的对照，再对代表性输入做匹配的随机初始化实验。这样既能判断输入表示差异，也能判断 ImageNet 预训练究竟提供了多少 cross-subject 增益。若 Raw Depth 单通道与三通道方案接近，只保留更简单、稳定的一种进入阶段 1C。

Depth 输入形式的最终决策必须依据：多折 subject-disjoint 平均 Accuracy、最差 fold、每类表现、模型大小和推理成本。`Depth_Color + ImageNet 预训练 ResNet18` 是最容易建立的起点；Raw Depth 是必须正式验证的候选；路线文档阶段不锁定最终主视觉输入。

### 4.3 阶段 1C：时序建模对照

在阶段 1A 的简单基线稳定，并完成阶段 1B 的输入对照后，再比较：

1. ResNet18 + mean pooling；
2. ResNet18 + attention pooling；
3. ResNet18 + GRU；
4. ResNet18 + 轻量 TCN；
5. ResNet18 + TSM；
6. ResNet18 + TSM + attention pooling。

默认先在阶段 1B 表现最好的 Depth 输入上比较时序模块；如果 Raw Depth 与 `Depth_Color` 多折结果非常接近，则二者都保留至少一个强时序实验，避免输入形式与时序模块之间存在交互而被遗漏。

TSM 是第一阶段非常值得优先验证的轻量视频建模方案。它通过在相邻帧间移动部分通道进行中层时序交互，几乎不增加参数和计算量；论文报告其能在保持 2D CNN 复杂度的同时引入有效时序建模：[TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)。

但 TSM 不是未经实验就确定的首发或最终结构。只有先建立 mean pooling 基线，才能判断 TSM、TCN、GRU 和 attention 带来了多少真实提升。最终时序模块由多折 subject-disjoint 结果决定。

对于候选 `ResNet18 + TSM + attention pooling`，可将输出组织为：

```text
[B,T,512]
→ LayerNorm
→ 时间注意力打分
→ 加权求和得到 [B,512]
→ 40 类
```

attention pooling 参数少于完整 Transformer，但仍必须与 mean pooling 对照。GRU 用于检验顺序依赖，TCN 用于检验多尺度局部时间模式，TSM 用于检验中层时序交互；这些候选不是为了全部堆叠保留。

帧数后续只比较 `8/12/16` 等少量离散配置，不把帧数作为连续超参数反复搜索。

### 4.4 共享训练配置

建议起点，不应视为固定答案：

```text
optimizer: AdamW
backbone_lr: 1e-4
head_lr: 5e-4 ~ 1e-3
weight_decay: 1e-4
epochs: 30 ~ 50
scheduler: cosine decay + 2~3 epoch warmup
loss: CrossEntropy + label_smoothing=0.05
precision: AMP
early_stop_metric: subject-disjoint validation Accuracy
```

使用预训练权重时，前 1–2 个 epoch 可以冻结 backbone，仅训练分类头或新增时序层，随后全部解冻。若显存允许，4070 Laptop 目标 batch 为 8–16 clips，3090 目标 batch 为 16–32 clips；不足时使用梯度累积。

初始 Demo 暂时不要加入 class weights、Focal Loss、SupCon 和复杂采样器。先得到干净基线，再逐项验证。

### 4.5 必做对照实验

按优先顺序：

1. `Depth_Color + ImageNet ResNet18 + mean pooling` 最小基线；
2. Raw Depth 单通道 + ResNet18 + mean pooling；
3. Raw Depth 三通道 + ResNet18 + mean pooling；
4. 代表性输入的 ImageNet 预训练 vs 随机初始化；
5. 最佳 Depth 输入 + attention pooling；
6. 最佳 Depth 输入 + GRU；
7. 最佳 Depth 输入 + 轻量 TCN；
8. 最佳 Depth 输入 + TSM；
9. 最佳 Depth 输入 + TSM + attention pooling；
10. 若 Raw Depth 与 `Depth_Color` 接近，在两者上复核最有希望的时序模块。

这组实验先回答“哪一种深度表示更适合跨用户”，再回答“哪一种时间建模真正有效”。不能同时更换输入、初始化、采样和时序模块后，只凭一个总分判断原因。

### 4.6 第二候选：X3D-XS 或 MoViNet-A0

若上述 ResNet18 时序对照已经稳定，且端到端 3D 建模仍有提升空间，再使用阶段 1B 选出的主视觉输入比较：

```text
最佳 Depth 表示 → X3D-XS → 40 类
最佳 Depth 表示 → MoViNet-A0 → 40 类
```

- X3D 通过联合扩展空间、时间、宽度和深度，在视频识别中提供良好的精度/计算量折中：[X3D](https://arxiv.org/abs/2004.04730)。
- MoViNet 面向移动端视频识别，具有小型 3D CNN 和流式推理设计：[MoViNets](https://openaccess.thecvf.com/content/CVPR2021/html/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.html)。

两者代码适配和调参成本高于简单 ResNet18 基线，因此属于后续候选，不是首日 Demo，也不预设一定优于 TSM/TCN/GRU。

### 4.7 阶段退出条件

- 一条命令完成训练与验证；
- 一条命令对 405 个测试 clip 生成格式正确的 CSV；
- `Depth_Color + ResNet18 + mean pooling` 在固定 `dev_fold` 上重复两次结果基本稳定；
- 完成 Raw Depth 单通道、Raw Depth 三通道与 `Depth_Color` 的正式对照；
- 完成代表性输入的 ImageNet 预训练 vs 随机初始化对照；
- 完成 mean pooling、GRU/TCN、TSM 等主要时序路线的比较；
- 用多折 subject-disjoint 结果确定暂定主视觉输入和时序候选；
- 保存每类混淆矩阵与错误 clip ID，但不人工标注测试集；
- 得到第一份有效 Kaggle submission；
- 模型大小和推理耗时已记录。

---

## 5. 阶段 2：Skeleton 基线与第一次多模态融合

在阶段 1 完成 `Depth_Color`/Raw Depth 与时序模块对照后，Skeleton 应作为第一个重点辅助模态，而不是先继续堆三个视觉模态。下文中的 Depth 分支指阶段 1 通过多折实验选出的 Raw Depth 或 `Depth_Color` 方案。Skeleton 与 Depth 的信息类型互补：

- Depth 擅长物体、房间和人体外观；
- Skeleton 更直接描述人体姿态与运动，对背景、衣着和光照不敏感；
- `Stand_up/Sit_down/Lie_down/Walk/Squat/Lunge/Jumping Jack` 等类别预计更依赖骨架动态；
- `Eat/Pour/Stir/Peel/Take medicine/Take temperature` 等物体交互动作更依赖视觉。

### 5.1 Skeleton 数据审计

解压后先确定：

- 关节数和关节顺序；
- 2D/3D 坐标与置信度字段；
- 是否存在多人、丢帧、全零关节和身份切换；
- `visualizations/` 是否只是可视化，训练必须优先使用原始数值数据；
- Skeleton 与 Depth 的长度和时间戳如何对应。

### 5.2 归一化

推荐依次加入：

1. 以骨盆/髋中心为原点做平移归一化；
2. 以肩宽或躯干长度做尺度归一化；
3. 使用肩线/髋线做适度朝向归一化；
4. 对缺失关节保留 validity mask；
5. 将序列插值到固定长度，如 64 或 96；
6. 构造坐标、速度差分、骨骼向量三类输入。

不能只输入绝对坐标，否则模型容易记住人的体型、站位和摄像机位置。

### 5.3 首选 Skeleton 模型

```text
坐标 + 速度 + 骨骼向量
→ 关节维投影
→ 多尺度轻量 TCN（kernel 3/5/7，depthwise separable）
→ 时间 attention pooling
→ 40 类
```

推荐先用轻量 TCN，而不是直接上复杂 GCN/大型 Transformer：

- 参数量可以控制在 1–3M；
- 容易处理不同长度和 validity mask；
- 对人体动作的局部与多尺度时间变化建模直接；
- 与视觉模型的工程接口简单；
- 小数据条件下比大型 Transformer 更不容易过拟合。

增强包括小幅 3D 旋转、尺度扰动、时间裁剪、时间缩放、关节噪声与少量关节 dropout；不能做会改变动作语义的强增强。

可作为第二候选的纯 CNN 骨架结构是 Ta-CNN。它证明了普通卷积也能有效建模骨架拓扑，并具有较低复杂度：[Topology-Aware CNN](https://ojs.aaai.org/index.php/AAAI/article/view/20191)。若骨架数据质量好且 TCN 接近瓶颈，再评估轻量 Skeleton Transformer，例如 UniSTFormer 风格的统一时空注意力；第一版无需实现。

### 5.4 第一次融合：先 logits，后 feature

#### 方案 A：后融合，必须先做

```text
Depth model logits ─┐
                    ├→ α·Depth + (1-α)·Skeleton → 40 类
Skeleton logits ────┘
```

在 OOF 预测上选择融合权重 `α`，不要在 Kaggle 公榜上选。后融合几乎没有开发风险，能快速回答最关键的问题：Skeleton 是否真的弥补了 Depth 的错误。

#### 方案 B：门控特征融合，作为本阶段主模型

```text
Depth embedding [512] → projection [256] ─┐
                                           ├→ gate → fused [256] → 40 类
Skeleton embedding     → projection [256] ─┘
```

gate 根据两个 embedding、模态存在 mask 和简单质量指标生成权重。质量指标可包含有效帧比例、有效关节比例、Skeleton 平均置信度等。

训练顺序：

1. 分别训练并冻结两个单模态 encoder；
2. 只训练 projection、gate、fusion head；
3. 用较小学习率端到端微调；
4. 同时保留两个单模态辅助分类头，避免某个分支在联合训练中退化。

总损失起点：

```text
L = L_fused + 0.2·L_depth + 0.2·L_skeleton
```

### 5.5 阶段退出条件

- Skeleton 单模态跨用户结果稳定；
- 完成 Depth/Skeleton 后融合，并用 OOF 证明两者具有互补性；
- 门控特征融合的平均 Accuracy 超过最佳单模态；
- 最差 fold 不因融合明显退化；
- 分析主要获益和退化的类别。

如果 Skeleton 不提升，不要直接删除。先区分是：数据解析错误、归一化错误、时间未对齐、单模态过弱，还是信息确实冗余。

---

## 6. 阶段 3：依次加入其余模态

不要同时加入四种模态。每次只新增一个模态，并完成“单模态基线 → OOF 后融合 → 特征融合 → 消融”。推荐顺序如下。

### 6.1 IR：第三个优先模态

IR 与 Depth 都是视觉模态，开发成本低，可以复用相同的帧采样、TSM 和训练框架。

候选：

```text
IR → ImageNet 预训练 ResNet18/MobileNetV3 + TSM → embedding/logits
```

若单独增加一套 ResNet18 导致最终模型过大，比较：

1. Depth 与 IR 共享 ResNet18 主干，使用模态专属 stem/BN/adapter；
2. Depth 使用 ResNet18，IR 使用 MobileNetV3-Small；
3. 训练独立 IR 教师，但最终把其知识蒸馏进共享视觉学生。

加入 IR 的理由不是“它也是图像所以一定有用”，而是检验它是否能在遮挡、深度伪彩不稳定或特定物体交互上补充 Depth。

### 6.2 IMU：高互补性的运动模态

IMU 单模态 Accuracy 可能不如视觉，但它不依赖摄像机背景，可能显著帮助运动类动作和跨环境泛化。

解压后首先确定五个设备的位置、每个 CSV 的时间戳、加速度/角速度/姿态角字段和采样率。

首选：

```text
五位置 IMU
→ 每设备时间对齐/重采样
→ 每通道训练集统计量标准化
→ 多尺度 1D depthwise CNN/TCN
→ 设备位置 embedding
→ 1–2 层小型 Transformer 或 attention pooling
→ embedding/logits
```

增强：小幅高斯噪声、幅值缩放、时间缩放、局部时间 masking。轴旋转增强只有在明确传感器坐标系后才能加入。

### 6.3 Thermal：有价值但要考虑缺失率

Thermal 可复用视觉框架，但可能与 IR/Depth 高度冗余，而且部分样本缺失。先做独立单模态和 OOF 后融合，确认其边际收益。

如果收益集中在少数类别，门控融合应允许模型按 clip/类别动态使用 Thermal，而不是固定平均。视觉增强只能做轻微强度与对比度变化，不应套用自然 RGB 的强颜色增强。

### 6.4 Radar：最后开发，优先保证解析正确

Radar 可能具有强运动互补性，但文件解析和表示方式最不确定，应放在数据管线成熟后开发。

根据实际 CSV 字段选择：

- 若是点云：构造 `x/y/z/velocity/intensity`，按时间重采样并做点数 mask；
- 若能形成 range-angle / range-Doppler 图：优先栅格化后用小型 2D CNN；
- 若每帧点数不定：可做统计池化 + 1D TCN，或在规则确认允许的情况下评估 PointNet 类结构；
- 必须记录空帧、异常点和坐标范围。

推荐的合规保守方案：

```text
Radar CSV
→ 空间/速度栅格化
→ 小型 2D CNN 提取每时刻特征
→ TCN/attention pooling
→ embedding/logits
```

### 6.5 模态保留标准

某个模态进入最终模型，至少满足以下一项：

- 跨用户 CV 平均 Accuracy 稳定提升；
- 最差 fold 明显改善；
- 对关键混淆类别有稳定改善；
- 在其他模态缺失时明显提高鲁棒性；
- 对全新环境更稳定，且模型大小/延迟成本合理。

单模态分数低不等于没有融合价值；反过来，单模态分数高也不等于值得保留，因为它可能与 Depth 完全重复。

---

## 7. 阶段 4：缺失模态鲁棒融合与跨受试者优化

### 7.1 先建立缺失模态评测矩阵

对每个正式模型都测试：

```text
当前保留的全部模态
缺 Thermal
缺 Radar
缺 IR
仅 Depth + Skeleton
仅视觉模态
仅运动模态
按训练/测试实际缺失模式采样
```

不能只报告“全部模态都存在”的结果。

### 7.2 Modality Dropout

联合训练时随机丢弃完整模态，而不是只把部分元素设零：

```text
训练输入 = 原始可用模态 × 随机 modality mask
```

丢弃概率应参考真实缺失率，并保证至少保留一个模态。建议先对 Thermal/Radar 设置较高 dropout，再做统一 dropout 的对照。

### 7.3 门控融合主线

门控融合输入：

- 各模态 embedding；
- 模态类型 embedding；
- modality mask；
- 有效帧率、有效关节率、雷达点数等质量特征。

输出每个模态的可靠性权重，再做加权融合。门控模型参数少、可解释、容易诊断，是进入 Transformer 前必须建立的强基线。

### 7.4 小型 Transformer 融合候选

当模态达到 3 个以上且门控融合稳定后，比较：

下图列出全部候选 token 只是为了说明融合接口；实际输入只包含当时已通过单模态、后融合和消融验证的模态，不要求六种模态同时存在。

```text
[CLS]
[Depth token]
[IR token]
[Thermal token]
[Skeleton token]
[IMU token]
[Radar token]
→ 2 层 Transformer Encoder
→ CLS → 40 类
```

建议起点：`d_model=192/256`、`n_heads=4`、`layers=2`。缺失模态使用 attention mask，并加入模态类型和质量 embedding。

选择 Transformer 的理由：它能建模“某一模态在当前 clip 中是否应该信任”以及模态间的条件关系。相关研究发现，在缺失模态动作识别中，Transformer 融合通常比简单求和或拼接更鲁棒，并提出随机丢模态再重建的 ActionMAE：[Towards Good Practices for Missing Modality Robust Action Recognition](https://arxiv.org/abs/2211.13916)。

但 Transformer 不保证必然优于门控融合：训练样本有限时容易过拟合。因此最终选择必须看跨用户多折结果，而不是模型是否“更先进”。

### 7.5 ActionMAE/特征重建分支

如果 Modality Dropout 有效，可以进一步尝试：

- 随机遮掉一个模态 embedding；
- 通过其余模态预测被遮掉的 embedding；
- 同时完成分类和特征重建。

损失示例：

```text
L = L_cls + λ_rec · L_reconstruct
```

这一分支应在简单 mask-aware Transformer 已稳定后再做，因为重建目标过强可能迫使不同模态丢失自身特有信息。

### 7.6 跨受试者优化顺序

1. 正确的用户级验证；
2. 人体/传感器的物理合理归一化；
3. 每模态合理增强；
4. 类别不平衡处理；
5. Supervised Contrastive Learning；
6. 跨模态同 clip 对比学习；
7. 必要时再考虑域对抗或用户身份去相关。

类别不平衡应依次比较：普通 CE、balanced sampler、Balanced Softmax/重加权 CE、Focal Loss。主指标是 Accuracy，过强的类别均衡可能提高 Macro-F1 却降低总 Accuracy，因此不能盲目使用。

对比学习建议：

- 同一动作不同用户作为正样本，促进用户无关表示；
- 同一 clip 的不同模态作为跨模态正样本；
- 不同动作作为负样本；
- 对容易混淆的细粒度动作避免过强地压缩类内差异。

---

## 8. 阶段 5：蒸馏路线

蒸馏不是前期救命工具，而是在强教师和可靠验证已经建立后，把性能压入 100 MB 模型的手段。

### 8.1 最推荐：已验证的多模态教师 → 轻量多模态学生

教师可以使用更宽的已验证模态 encoder、更强融合和更多训练视图；学生使用最终部署结构。这里的“多模态教师”不表示默认使用全部六种模态，只使用前面实验已经证明有价值的模态组合。

基础损失：

```text
L_student = L_CE
          + λ_kd · KL(student_logits/T, teacher_logits/T)
          + λ_feat · MSE(P(student_feature), teacher_feature)
```

温度 `T` 可从 2–4 开始。先做 logit KD，再判断是否需要 feature KD。

### 8.2 跨模态蒸馏

可能路线：

1. **Full-to-partial KD**：全模态教师指导随机缺失模态的学生，提升缺失模态鲁棒性。
2. **Visual-to-motion KD**：强视觉教师指导 Skeleton/IMU 分支学习类别边界。
3. **Motion-to-visual KD**：Skeleton/IMU 的运动知识指导视觉分支，减少对背景的依赖。
4. **Multi-teacher KD**：Depth、Skeleton、IMU 独立教师共同指导统一学生。

对动作识别，除了 logits，还可以蒸馏时间关系矩阵、帧间相似度和 attention 分布。多模态蒸馏在动作识别中常用于把多流知识压入更轻的推理模型，可参考：[Multimodal Distillation for Egocentric Action Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Radevski_Multimodal_Distillation_for_Egocentric_Action_Recognition_ICCV_2023_paper.html)。

### 8.3 自蒸馏与 EMA 教师

如果大教师收益不稳定，优先试成本更低的：

- EMA teacher；
- 同架构上一代 checkpoint 教当前模型；
- 深层分类头指导浅层辅助头；
- 同一模型强增强视图指导弱增强视图。

### 8.4 蒸馏的停止条件

- 学生跨用户平均分提升且最差 fold 不退化；
- 学生校准和缺失模态性能没有明显恶化；
- 教师—学生容量差距不能过大；
- 如果 KD 仅提升单一 fold 或公榜，则不进入最终方案。

---

## 9. 阶段 6：压缩、集成与部署

### 9.1 模型大小预算

建议最终 FP32 权重本身控制在 70–85 MB，而不是卡在 99.9 MB。下面是候选模块的预算示意，不表示所有分支一定同时进入最终模型：

```text
主视觉 encoder + 轻量时序模块 约 45–55 MB
Skeleton encoder             约 4–10 MB
IMU encoder                  约 2–6 MB
Radar encoder                约 2–6 MB
modality adapters + fusion   约 3–8 MB
余量                         约 10–20 MB
```

真实大小以序列化后的最终 checkpoint 为准，不只看参数统计。提交 checkpoint 不包含 optimizer、scheduler、EMA 历史和无用辅助头。

### 9.2 推荐压缩顺序

1. 共享视觉 encoder 或使用小型模态 adapter；
2. 蒸馏；
3. 删除训练期辅助头；
4. FP16 权重/推理（规则与运行环境确认后）；
5. INT8 量化；
6. 结构化剪枝。

量化和剪枝放后面，因为它们的开发与校准成本更高，而且跨模态模型可能对量化误差敏感。

### 9.3 集成策略

优先级：

1. 单模型 EMA；
2. 同架构多 fold/多 seed 的权重平均或 model soup，最终仍是一个 checkpoint；
3. 在总大小合规时使用少量 logits ensemble；
4. 轻量 temporal TTA，例如两个时间采样视图。

模型权重平均前必须保证架构相同，并在 OOF 上验证。TTA 必须同时记录准确率增益与推理时间。

### 9.4 推理包

从中期开始维护：

```text
README.md
requirements.txt / environment.yml
configs/final.yaml
checkpoints/model.pth
inference.sh
src/infer.py
```

`inference.sh` 接收数据根目录和输出 CSV 路径；能处理任意存在/缺失的模态；不依赖训练目录；在干净环境中可从零复现。

---

## 10. 推荐实验顺序与决策表

| 顺序 | 实验 | 主要回答的问题 | 通过后进入下一步 |
|---|---|---|---|
| 1 | `Depth_Color + ResNet18 + mean pooling` | 数据、标签、划分和完整闭环是否正确？ | 最小基线稳定 |
| 2 | Raw Depth 单通道/三通道 vs `Depth_Color` | 哪种深度表示更适合跨用户？ | 多折完成输入选择 |
| 3 | ImageNet 预训练 vs 随机初始化 | 预训练是否带来真实泛化收益？ | 结论跨 fold 一致 |
| 4 | attention pooling、TCN、GRU | 后期时序建模是否有效？ | 优于 mean pooling |
| 5 | ResNet18 + TSM / TSM + attention | 中层时序交互是否更好？ | 多折稳定 |
| 6 | Skeleton TCN | 运动/姿态单模态有多强？ | 数据解析可靠 |
| 7 | Depth + Skeleton logits 后融合 | 两模态是否互补？ | OOF 提升 |
| 8 | Depth + Skeleton 门控特征融合 | 联合学习是否优于后融合？ | 多折提升 |
| 9 | 加 IR | 额外视觉信息是否有边际收益？ | 性价比合理 |
| 10 | 加 IMU | 是否改善运动类和跨环境泛化？ | 最差 fold 改善 |
| 11 | 加 Thermal | 收益是否超过缺失/模型成本？ | 有稳定边际收益 |
| 12 | 加 Radar | 雷达表示是否带来互补运动信息？ | 解析与收益都稳定 |
| 13 | Modality Dropout | 缺模态时是否稳定？ | 缺失矩阵改善 |
| 14 | 小型 Transformer 融合 | 是否优于门控？ | 多折证据充分 |
| 15 | SupCon/跨模态对比 | 是否改善跨用户表征？ | 不损害细粒度类别 |
| 16 | 多模态蒸馏 | 能否把教师性能压进小模型？ | 学生稳定提升 |
| 17 | 权重平均/量化/TTA | 最终性能—效率折中如何？ | 满足提交要求 |

任何一步未通过，都应停下来分析原因，而不是继续把后面的复杂模块叠上去。

---

## 11. 团队并行分工建议（这部分先不用看）

### 成员 A：数据与视觉分支

- 解压/manifest/图像解码；
- Depth、IR、Thermal；
- ResNet18 + TSM、X3D/MoViNet 对照；
- 视觉增强与帧采样。

### 成员 B：运动传感器分支

- Skeleton 解析、归一化和 TCN；
- IMU 对齐、重采样和 1D 模型；
- Radar 解析和栅格化；
- 各模态数据质量指标。

### 成员 C：验证、融合与部署

- subject-disjoint folds 与统一评测；
- OOF 后融合、门控融合、Transformer；
- 模态 dropout、蒸馏与消融；
- submission、模型大小、推理脚本和复现环境。

三人不能各自维护不兼容的数据格式和训练脚本。所有分支必须基于统一 manifest、统一 sample dict、统一配置与统一评测器。

Agent 适合处理模板代码、单元测试、配置生成、日志解析和重复性重构；涉及数据语义、验证划分、增强合法性、测试集使用方式和最终模型决策必须由队员复核。

---

## 12. 从 7 月 14 日到 9 月 15 日的建议节奏（只做参考也不用看）

### 第 1 周：地基与 Depth Demo

- 解压、审计、manifest；
- 固定跨用户 folds；
- 跑通 `Depth_Color + ResNet18 + mean pooling` 最小基线；
- 建立 Raw Depth 单通道/三通道输入并完成初步对照；
- 完成第一份有效 submission。

### 第 2 周：时序模型与 Skeleton

- 完成 Raw Depth、`Depth_Color` 及预训练/随机初始化的多折对照；
- TCN/GRU/TSM 对照；
- Skeleton 解析、归一化、轻量 TCN；
- Depth/Skeleton OOF 后融合。

### 第 3 周：第一次联合模型

- 门控特征融合；
- 分支辅助损失；
- IR 与 IMU 单模态并行开发。

### 第 4 周：扩展模态

- 依次加入 IR、IMU；
- Thermal、Radar 完成可用基线；
- 形成模态边际收益表。

### 第 5 周：鲁棒融合

- modality mask/dropout；
- 门控与小型 Transformer 对照；
- 缺失模态评测矩阵。

### 第 6 周：跨用户与长尾优化

- 增强、采样/损失、SupCon；
- 多折正式验证；
- 固定最终模态组合。

### 第 7 周：蒸馏与压缩

- 基于已验证模态组合建立强教师；
- logit KD、feature KD、full-to-partial KD；
- 控制模型大小和延迟。

### 第 8 周：收敛与复现

- 多 fold/seed 训练；
- EMA/权重平均；
- 干净环境复现；
- 技术报告同步撰写。

### 最后一周：冻结

- 不再大改数据与模型结构；
- 确认两份最终候选；
- 完整检查 submission、checkpoint、README、inference.sh；
- 准备线上复现和现场新数据的异常处理。

---

## 13. 近期学习顺序（wcy's）（不用看）

为了能尽快参与实现，前置知识按下面顺序学习：

1. PyTorch `Dataset/DataLoader`、AMP、checkpoint、训练/验证循环；
2. 视频 tensor 组织 `[B,T,C,H,W]` 与分段采帧；
3. ResNet18、TSM、TCN、GRU 的输入输出和区别；
4. Group/subject-disjoint cross validation 与 OOF；
5. Skeleton 的中心化、尺度、骨骼向量、速度特征；
6. logits 后融合、feature fusion、门控；
7. modality mask/dropout 与 Transformer attention mask；
8. 知识蒸馏的 CE、KL、temperature 和 feature matching；
9. 模型参数量、checkpoint 大小和推理 profiling。

近期最重要的是掌握前四项。蒸馏和复杂 Transformer 可以在基础模型稳定后再深入。

---

## 14. 第一轮应立即创建的任务清单

- [ ] 使用 7-Zip 解压训练与测试数据；
- [ ] 验证 40 类、405 个测试 clip、各候选模态目录及 Raw Depth/`Depth_Color` 两种表示；
- [ ] 生成统一 manifest；
- [ ] 输出用户/类别/模态缺失统计；
- [ ] 固定 `dev_fold` 和正式 `cv_folds`；
- [ ] 建立统一配置、日志和评测框架；
- [ ] 跑通随机预测 submission；
- [ ] 实现 `Depth_Color` 8 帧 Dataset；
- [ ] 实现 `Depth_Color + ImageNet ResNet18 + mean pooling` 最小基线；
- [ ] 实现 Raw Depth 单通道和三通道 Dataset；
- [ ] 完成 Raw Depth 单通道、Raw Depth 三通道与 `Depth_Color` 对照；
- [ ] 完成代表性 Depth 输入的 ImageNet 预训练 vs 随机初始化对照；
- [ ] 完成 attention pooling、TCN、GRU、TSM 等时序对照；
- [ ] 将 `ResNet18 + TSM + attention pooling` 作为强候选验证，而不是默认最终结构；
- [ ] 生成第一份本地 OOF、混淆矩阵与有效 Kaggle submission；
- [ ] 再启动 Skeleton 分支。

---

## 15. 当前路线的核心判断

1. **第一优先级不是寻找一个名字最“新”的 SOTA，而是建立不会泄漏用户信息的验证体系。**
2. **最稳妥的首个可运行视觉基线是 ImageNet 预训练 ResNet18 加时间 mean pooling；ResNet18 + TSM 是第一阶段优先验证的强时序候选，而不是未经对照实验就锁定的最终结构。**
3. **Raw Depth 与 `Depth_Color` 必须在相同 subject-disjoint 条件下正式比较，最终主视觉输入由多折结果决定。**
4. **第二模态优先 Skeleton，因为它与 Depth 的互补性比继续增加相似视觉流更值得验证。**
5. **所有模态都进入研究候选池，但最终模型不默认使用全部六种模态；只保留通过单模态、OOF 后融合和消融证明具有稳定边际收益的 2～4 种模态。**
6. **简单后融合是必做基线；门控特征融合是中期主线；小 Transformer 是有条件的升级项。**
7. **缺失模态要在训练阶段主动模拟，不能等测试时再零填充处理。**
8. **蒸馏必须建立在稳定学生、强教师和多折证据上，主要目标是提升轻量学生和缺失模态鲁棒性。**
9. **最终模型要为现场新用户与新环境优化，而不是只为当前公开榜优化。**

这份 `route_1` 的任务是让项目有序启动。数据审计完成后，应根据真实模态格式、缺失率、单模态结果和官方规则更新为 `route_2`，届时再锁定最终架构、参数预算和详细实验表。
