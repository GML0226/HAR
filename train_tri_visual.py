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
from src.tri_visual_data import TriVisualDataset
from src.tri_visual_model import TriVisualGatedClassifier
from src.utils import ensure_dir, load_config, save_json, seed_everything


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def encoder_stage(epoch: int, freeze_epochs: int, last_stage_epochs: int) -> str:
    if epoch <= freeze_epochs:
        return "frozen"
    if epoch <= freeze_epochs + last_stage_epochs:
        return "last_stage"
    return "full"


def compute_loss(output: dict[str, torch.Tensor], labels: torch.Tensor, original_mask: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    auxiliary = []
    for index in range(3):
        available = original_mask[:, index].bool()
        if available.any():
            auxiliary.append(criterion(output["auxiliary_logits"][available, index], labels[available]))
    auxiliary_loss = torch.stack(auxiliary).mean() if auxiliary else torch.zeros((), device=labels.device)
    return criterion(output["logits"], labels) + 0.2 * auxiliary_loss


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    labels, logits, clip_ids, user_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True):
            clip_ids.extend(batch["clip_id"])
            user_ids.extend(batch["user_id"])
            batch = move_batch(batch, device)
            output = model(batch["frames"], batch["modality_mask"], batch["quality"])
            labels.extend(batch["label"].cpu().tolist())
            logits.append(output["logits"].cpu().numpy())
    logits_array = np.concatenate(logits)
    labels_array = np.asarray(labels)
    predictions = logits_array.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels_array, predictions)), "macro_f1": float(f1_score(labels_array, predictions, average="macro", zero_division=0)), "samples": len(labels_array)}, labels_array, logits_array, clip_ids, user_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tri_visual_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    output_dir = ensure_dir(config["output_dir"])
    shutil.copy2(args.config, output_dir / "config.yaml")
    manifest = pd.read_csv(config["data"]["train_manifest"]).fillna("")
    train_frame, validation_frame = split_by_users(manifest, config["split"]["validation_users"])
    if set(train_frame.user_id) & set(validation_frame.user_id):
        raise RuntimeError("Subject leakage detected")
    data = config["data"]
    train_set = TriVisualDataset(train_frame, data["image_size"], data["frames"], training=True)
    validation_set = TriVisualDataset(validation_frame, data["image_size"], data["frames"], training=False)
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": config["training"]["num_workers"], "pin_memory": torch.cuda.is_available()}
    if loader_args["num_workers"] > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = TriVisualGatedClassifier(num_classes=data["num_classes"], **config["model"]).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()
    best_accuracy, stale, history = -1.0, 0, []
    for epoch in range(1, config["training"]["epochs"] + 1):
        stage = encoder_stage(epoch, config["training"]["freeze_epochs"], config["training"]["last_stage_epochs"])
        model.set_encoder_stage(stage)
        model.train()
        start = time.perf_counter()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} training", leave=False, dynamic_ncols=True):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                output = model(batch["frames"], batch["modality_mask"], batch["quality"], config["training"]["modality_dropout"])
                loss = compute_loss(output, batch["label"], batch["modality_mask"], criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics, labels, logits, clip_ids, user_ids = evaluate(model, validation_loader, device)
        metrics.update({"epoch": epoch, "stage": stage, "train_loss": float(np.mean(losses)), "seconds": round(time.perf_counter() - start, 1)})
        history.append(metrics)
        print(metrics, flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = metrics["accuracy"], 0
            torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, output_dir / "best.pt")
            np.savez_compressed(output_dir / "validation_logits.npz", logits=logits, labels=labels, clip_ids=np.asarray(clip_ids), user_ids=np.asarray(user_ids), prediction_source=np.asarray("development_validation"))
        else:
            stale += 1
            if stale >= config["training"]["early_stopping_patience"]:
                break
    checkpoint = output_dir / "best.pt"
    size_mb = checkpoint.stat().st_size / 1024 / 1024
    if size_mb > 100:
        raise RuntimeError(f"Checkpoint exceeds 100 MB: {size_mb:.3f} MB")
    save_json({"history": history, "best_accuracy": best_accuracy}, output_dir / "metrics.json")
    save_json({"parameters": int(sum(parameter.numel() for parameter in model.parameters())), "checkpoint_megabytes": round(size_mb, 3), "under_80_mb_target": size_mb <= 80}, output_dir / "model_profile.json")


if __name__ == "__main__":
    main()
