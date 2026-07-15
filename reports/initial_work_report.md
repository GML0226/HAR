# CUHK-X Small Model Track 初步工作报告

日期：2026-07-15

## 一、任务范围与规则结论

本赛题是 40 类日常动作识别，核心挑战是跨受试者泛化：训练用户为 1-9、16-24，线上测试用户为未见过的 10-11、25-26。最终提交的序列化模型必须不大于 100 MB，网络结构限于 CNN、RNN、Transformer 及其常规组合。

根据主办方在讨论区的正式澄清，标准的 ImageNet 预训练 ResNet18（约 44 MB）允许用于 Small Model Track；知识蒸馏也允许。因此，本项目将 ImageNet 预训练 ResNet18 作为阶段 1 的合规候选模型。大型基础模型不在本项目范围内。

## 二、整体技术路线

1. 阶段 0：完成数据审计、跨受试者划分和训练至提交的完整流程验证。
2. 阶段 1A：建立 `Depth_Color -> ImageNet ResNet18 -> 时间平均池化 -> 40 类` 的视觉基线。
3. 阶段 1B：在完全相同的划分和训练条件下比较 ImageNet 预训练与随机初始化；仅在提供 Raw Depth 数据后再比较深度表示。
4. 阶段 1C：在选定的深度表示上比较 mean pooling、attention pooling、GRU、TCN、TSM。
5. 阶段 2：训练做人体归一化的 Skeleton TCN，并通过 OOF 预测验证其与视觉模型的后融合价值。
6. 阶段 3：按 IR、IMU、Thermal、Radar 的顺序逐个加入模态；每个模态必须先证明单模态能力和 OOF 融合增益。
7. 阶段 4 及以后：缺失模态鲁棒训练、紧凑门控或 Transformer 融合、蒸馏、多折验证和最终模型大小审计。

## 三、已完成工作

### 3.1 数据与验证地基

- 已生成 `artifacts/manifests/train_manifest.csv` 与 `artifacts/manifests/test_manifest.csv`。
- 已审计 3,036 个训练 clip、405 个测试 clip 和 40 个类别。
- 已确认训练数据中有 18 位用户。固定开发验证用户为 `user3`、`user5`、`user19`、`user24`；它们来自两个训练用户组，共 670 个 clip，覆盖全部 40 类。
- 训练和验证严格按用户隔离。训练脚本保存实际使用的用户列表，并在发现训练/验证用户重叠时立即报错。
- 已确认真实的缺失模态：测试集有 10 个 clip 缺 Thermal、1 个 clip 缺 Radar。现有六模态 Dataset 通过 modality mask 表示可用性。
- 官方示例提交文件中的 `prediction` 列不会被读取；仅使用其 `path` 顺序做提交校验。

### 3.2 六模态数据管线 Smoke Test

- 已实现包含 Depth_Color、IR、Thermal、Skeleton、IMU、Radar 的紧凑 CNN/TCN 门控融合 smoke 模型。
- 已在 CUDA 上完成 32 个训练样本、16 个验证样本、1 个 epoch 的 smoke 训练。
- 已生成格式正确的 405 行提交文件：`outputs/smoke/submission.csv`。
- 该 smoke run 的验证 Accuracy 为 0.0，不具有模型性能含义。它只证明数据解析、训练、权重保存、推理和 CSV 导出的链路完整可用。

### 3.3 阶段 1A：预训练 ResNet18 视觉基线

- 已新增独立的视觉数据读取路径，使阶段 1A 不会受到多模态解析、缺失模态或融合策略的影响。
- 已实现 `Depth_Color -> 预训练 ResNet18 -> 逐帧特征 -> 时间平均池化 -> 40 类线性分类头`。
- 已新增两份配置：
  - `configs/resnet18_depth.yaml`：224 像素、8 帧、30 epoch、主干与分类头独立学习率、首 epoch 冻结主干。
  - `configs/resnet18_depth_smoke.yaml`：112 像素、4 帧、32/16 样本、1 个 epoch。
- 已在 CUDA 上完成预训练 ResNet18 smoke run。16 个验证样本的结果为 Accuracy 12.5%、Macro-F1 0.026；这是可训练性检查，不是可用于模型筛选的正式分数。
- 已生成 `outputs/resnet18_depth_smoke/submission.csv`。它有 405 行、预测均为合法整数标签，且 path 顺序与官方模板完全一致。
- ResNet18 checkpoint 有 11,197,032 个参数，大小为 42.785 MB，单独满足 100 MB 限制。

## 四、当前进度

| 工作项 | 状态 | 产物或证据 |
| --- | --- | --- |
| 数据 manifest 与审计 | 已完成 | `artifacts/manifests/audit_report.json` |
| 跨受试者开发划分 | 已完成 | 各输出目录下的 `split_report.json` |
| 六模态管线 smoke test | 已完成 | `outputs/smoke/` |
| 预训练 ResNet18 阶段 1A smoke test | 已完成 | `outputs/resnet18_depth_smoke/` |
| 完整 ResNet18 开发训练 | 待执行 | 已完成配置，尚未启动 |
| 预训练与随机初始化对照 | 待执行 | 需要配对完整实验 |
| Raw Depth 对照 | 暂时阻塞 | 当前数据中未找到 Raw Depth 目录 |
| Skeleton 基线与 OOF 融合 | 待执行 | 在阶段 1 模型选择后开始 |
| 正式多折 CV 和最终提交 | 待执行 | 在候选结构确定后开始 |

## 五、已发现问题与工程决策

1. **缺少 Raw Depth。** 当前训练目录只有 `Depth_Color`、`IR`、`Thermal`、`Skeleton`、`IMU`、`Radar`，没有独立 Raw Depth 目录。在确认数据位置或补充数据之前，不能声称完成了路线文档中的 Raw Depth 对照。
2. **不能为三种视觉模态各自使用一个 ResNet18。** 单个权重文件约为 42.8 MB；三个独立分支在加入传感器和融合头前就会超过 100 MB。后续视觉融合应使用共享 ResNet18 主干、模态专属归一化或轻量 stem，或仅保留有稳定增益的模态。
3. **Smoke 分数不参与模型选择。** 它们的样本数和训练轮数不足，正式结论必须基于完整固定划分实验，随后再做多折跨受试者验证。
4. **类别分布不均衡。** 类别 25 只有 12 个 clip，类别 36 有 365 个 clip。第一轮完整训练使用普通交叉熵，以建立可解释参照；采样器、类别权重与其他损失函数属于后续消融实验，并以 Accuracy 为主指标、Macro-F1 为诊断指标。
5. **测试集不用于标签选择。** 测试数据仅用于无标签推理和提交格式校验，不能用于选超参数、融合权重或任何形式的标签推断。

## 六、后续工作与验收标准

1. 使用 `configs/resnet18_depth.yaml` 在固定跨受试者划分上完成两次不同随机种子的完整训练。
2. 将相同配置中的 `pretrained` 改为 `false`，保持帧数、划分、分辨率、epoch 和优化器不变，完成随机初始化对照。
3. 比较 Accuracy、Macro-F1、逐类 Recall、模型大小和最差 fold；只有预训练在跨受试者结果上稳定提升，才保留它。
4. 训练 Skeleton TCN，保存视觉和骨架的 OOF 概率；通过 OOF logits 后融合判断 Skeleton 是否具有互补价值，不在 Kaggle 榜单上选择融合权重。
5. 只有在固定开发划分上稳定提升的候选，才进入 4-5 折 user-disjoint CV。
6. 每次提交前重新检查最终模型文件不大于 100 MB。

## 七、运行命令

```powershell
python scripts/build_manifest.py
python train_visual.py --config configs/resnet18_depth_smoke.yaml
python infer_visual.py --config configs/resnet18_depth_smoke.yaml --checkpoint outputs/resnet18_depth_smoke/best.pt --output outputs/resnet18_depth_smoke/submission.csv

# 阶段 1A 完整训练
python train_visual.py --config configs/resnet18_depth.yaml
python infer_visual.py --config configs/resnet18_depth.yaml --checkpoint outputs/resnet18_depth/best.pt --output outputs/resnet18_depth/submission.csv
```
