# CUHK-X Small Model Track Demo

The baseline is a compact six-modality CNN/TCN model trained from scratch. It uses subject-disjoint validation and does not read the `prediction` column in the supplied sample submission.

## Run

```powershell
python scripts/build_manifest.py
python train.py --config configs/smoke.yaml
python infer.py --config configs/smoke.yaml --checkpoint outputs/smoke/best.pt --output outputs/smoke/submission.csv
```

After the smoke test passes, use `configs/demo.yaml` for the full initial experiment.

## Phase 1A: Depth_Color + ImageNet ResNet18

The competition host explicitly allows standard ImageNet-pretrained ResNet18. This baseline follows the staged route: it samples Depth_Color frames uniformly, extracts each frame with ResNet18, mean-pools over time, and predicts 40 classes.

```powershell
python train_visual.py --config configs/resnet18_depth_smoke.yaml
python infer_visual.py --config configs/resnet18_depth_smoke.yaml --checkpoint outputs/resnet18_depth_smoke/best.pt --output outputs/resnet18_depth_smoke/submission.csv
```

Use `configs/resnet18_depth.yaml` only after the smoke run has passed.

For the optimized full run, `configs/resnet18_depth_optimized.yaml` caches frame paths once, uses four persistent data-loading workers, and trains with a batch size of 12. It writes separate artifacts to avoid mixing it with the earlier run.

python train_visual.py --config configs/resnet18_depth.yaml
python infer_visual.py --config configs/resnet18_depth.yaml --checkpoint outputs/resnet18_depth/best.pt --output outputs/resnet18_depth/submission.csv