from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import split_by_users
from src.utils import ensure_dir, load_config, save_json, seed_everything
from src.visual_data import FrameSequenceDataset
from src.visual_model import ResNet18MeanPool


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def report(message: str, log_path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def gpu_memory_mb(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    return f"allocated={allocated:.0f}MB reserved={reserved:.0f}MB"


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int, total_epochs: int) -> tuple[dict, list[int], list[int], list[str]]:
    model.eval()
    labels, predictions, clip_ids = [], [], []
    with torch.no_grad():
        progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} validation", leave=False, dynamic_ncols=True)
        for batch in progress:
            clip_ids.extend(batch["clip_id"])
            batch = move_batch(batch, device)
            prediction = model(batch["frames"], batch["frame_mask"]).argmax(dim=1)
            labels.extend(batch["label"].cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "samples": len(labels),
    }, labels, predictions, clip_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18_depth.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    output_dir = ensure_dir(config["output_dir"])
    shutil.copy2(args.config, output_dir / "config.yaml")
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    manifest = pd.read_csv(config["data"]["train_manifest"]).fillna("")
    train_frame, validation_frame = split_by_users(manifest, config["split"]["validation_users"])
    for name, limit in (("train", config["training"].get("max_train_samples")), ("validation", config["training"].get("max_validation_samples"))):
        if limit:
            frame = train_frame if name == "train" else validation_frame
            selected = frame.sample(min(int(limit), len(frame)), random_state=config["seed"]).sort_index()
            if name == "train":
                train_frame = selected
            else:
                validation_frame = selected
    train_users, validation_users = set(train_frame.user_id), set(validation_frame.user_id)
    if train_users & validation_users:
        raise RuntimeError("Subject leakage detected")
    save_json({"train_users": sorted(train_users), "validation_users": sorted(validation_users), "overlap": [], "train_samples": len(train_frame), "validation_samples": len(validation_frame)}, output_dir / "split_report.json")

    data = config["data"]
    train_set = FrameSequenceDataset(train_frame, data["modality"], data["image_size"], data["frames"])
    validation_set = FrameSequenceDataset(validation_frame, data["modality"], data["image_size"], data["frames"])
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": config["training"]["num_workers"], "pin_memory": torch.cuda.is_available()}
    if config["training"]["num_workers"] > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = ResNet18MeanPool(num_classes=data["num_classes"], **config["model"]).to(device, memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report(
        f"Device={device}; GPU={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}; "
        f"parameters={parameter_count:,}; pretrained={config['model']['pretrained']}",
        log_path,
    )
    report(
        f"Data: train={len(train_set)}, validation={len(validation_set)}, frames={data['frames']}, "
        f"resolution={data['image_size']}px, batch_size={config['training']['batch_size']}, "
        f"workers={config['training']['num_workers']}",
        log_path,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["training"]["backbone_learning_rate"]},
        {"params": model.classifier.parameters(), "lr": config["training"]["head_learning_rate"]},
    ], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy, stale, history = -1.0, 0, []
    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.set_backbone_trainable(epoch > config["training"]["freeze_backbone_epochs"])
        model.train()
        report(
            f"Epoch {epoch}/{config['training']['epochs']} started; backbone_trainable={epoch > config['training']['freeze_backbone_epochs']}; "
            f"lr_backbone={optimizer.param_groups[0]['lr']:.2e}; lr_head={optimizer.param_groups[1]['lr']:.2e}; {gpu_memory_mb(device)}",
            log_path,
        )
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} training", leave=False, dynamic_ncols=True)
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = loss_fn(model(batch["frames"], batch["frame_mask"]), batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(train_loader):
                progress.set_postfix(loss=f"{losses[-1]:.4f}", avg_loss=f"{np.mean(losses):.4f}", gpu=gpu_memory_mb(device).split()[0])
        scheduler.step()
        metrics, labels, predictions, clip_ids = evaluate(model, validation_loader, device, epoch, config["training"]["epochs"])
        metrics.update({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.perf_counter() - epoch_start, 1),
        })
        history.append(metrics)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = metrics["accuracy"], 0
            torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, output_dir / "best.pt")
            pd.DataFrame({"clip_id": clip_ids, "label": labels, "prediction": predictions}).to_csv(output_dir / "validation_predictions.csv", index=False)
            status = "new_best"
        else:
            stale += 1
            status = "no_improvement"
        report(
            f"Epoch {epoch}/{config['training']['epochs']} finished in {metrics['epoch_seconds']:.1f}s; "
            f"train_loss={metrics['train_loss']:.4f}; val_accuracy={metrics['accuracy']:.4f}; "
            f"macro_f1={metrics['macro_f1']:.4f}; best_accuracy={best_accuracy:.4f}; "
            f"status={status}; stale_epochs={stale}/{config['training']['early_stopping_patience']}; {gpu_memory_mb(device)}",
            log_path,
        )
        if status == "no_improvement":
            if stale >= config["training"]["early_stopping_patience"]:
                report("Early stopping triggered.", log_path)
                break
    checkpoint = output_dir / "best.pt"
    save_json({"history": history, "best_accuracy": best_accuracy}, output_dir / "metrics.json")
    save_json({"parameters": int(sum(item.numel() for item in model.parameters())), "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_megabytes": round(checkpoint.stat().st_size / 1024 / 1024, 3), "under_100_mb": checkpoint.stat().st_size <= 100 * 1024 * 1024, "pretrained": config["model"]["pretrained"], "device": str(device)}, output_dir / "model_profile.json")
    report(f"Training completed. Best checkpoint: {checkpoint}; best_accuracy={best_accuracy:.4f}", log_path)


if __name__ == "__main__":
    main()
