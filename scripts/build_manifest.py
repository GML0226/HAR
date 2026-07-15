from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

MODALITIES = ("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_action(action_dir: Path) -> tuple[int, str]:
    label, _, name = action_dir.name.partition("_")
    return int(label), name


def count_files(path: Path, modality: str) -> int:
    if not path.exists():
        return 0
    if modality in {"Depth_Color", "IR", "Thermal"}:
        return sum(item.suffix.lower() in IMAGE_SUFFIXES for item in path.rglob("*"))
    if modality == "Skeleton":
        return sum(item.suffix.lower() == ".json" for item in path.rglob("*.json"))
    return sum(item.suffix.lower() == ".csv" for item in path.rglob("*.csv"))


def collect_train(root: Path) -> list[dict]:
    records: dict[tuple[int, str, str], dict] = {}
    for modality in MODALITIES:
        modality_root = root / "Training" / "data" / modality
        if not modality_root.exists():
            continue
        for action_dir in modality_root.iterdir():
            if not action_dir.is_dir() or "_" not in action_dir.name:
                continue
            action_id, action_name = parse_action(action_dir)
            for user_dir in action_dir.iterdir():
                if not user_dir.is_dir() or not user_dir.name.startswith("user"):
                    continue
                for trial_dir in user_dir.iterdir():
                    if not trial_dir.is_dir():
                        continue
                    key = (action_id, user_dir.name, trial_dir.name)
                    record = records.setdefault(
                        key,
                        {
                            "clip_id": f"{action_id}_{user_dir.name}_{trial_dir.name}",
                            "action_id": action_id,
                            "action_name": action_name,
                            "user_id": user_dir.name,
                            "trial_id": trial_dir.name,
                        },
                    )
                    record[f"{modality}_path"] = str(trial_dir)
                    record[f"{modality}_length"] = count_files(trial_dir, modality)
    for record in records.values():
        for modality in MODALITIES:
            record.setdefault(f"{modality}_path", "")
            record.setdefault(f"{modality}_length", 0)
            record[f"has_{modality}"] = int(record[f"{modality}_length"] > 0)
    return sorted(records.values(), key=lambda item: (item["action_id"], item["user_id"], item["trial_id"]))


def collect_test(root: Path) -> list[dict]:
    test_root = root / "Testing" / "data" / "small_model_track_test"
    records: list[dict] = []
    for clip_dir in sorted(test_root.glob("SM_test_*")):
        if not clip_dir.is_dir():
            continue
        record = {
            "clip_id": clip_dir.name,
            "submission_path": f"small_model_track_test/{clip_dir.name}/",
        }
        for modality in MODALITIES:
            path = clip_dir / modality
            record[f"{modality}_path"] = str(path) if path.exists() else ""
            record[f"{modality}_length"] = count_files(path, modality)
            record[f"has_{modality}"] = int(record[f"{modality}_length"] > 0)
        records.append(record)
    return records


def audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    report = {
        "train_clips": len(train),
        "test_clips": len(test),
        "class_count": int(train.action_id.nunique()),
        "users": sorted(train.user_id.unique().tolist()),
        "clips_per_class": {str(key): int(value) for key, value in train.action_id.value_counts().sort_index().items()},
        "clips_per_user": {str(key): int(value) for key, value in train.user_id.value_counts().sort_index().items()},
        "modalities": {},
    }
    for modality in MODALITIES:
        report["modalities"][modality] = {
            "train_available": int(train[f"has_{modality}"].sum()),
            "test_available": int(test[f"has_{modality}"].sum()),
            "train_median_length": float(train[f"{modality}_length"].median()),
            "test_median_length": float(test[f"{modality}_length"].median()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="cuhk_dataset/Small-Model-Track")
    parser.add_argument("--output-dir", default="artifacts/manifests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(collect_train(Path(args.data_root)))
    test = pd.DataFrame(collect_test(Path(args.data_root)))
    if train.empty or test.empty:
        raise RuntimeError("No training or test clips found. Verify --data-root.")
    train.to_csv(output_dir / "train_manifest.csv", index=False)
    test.to_csv(output_dir / "test_manifest.csv", index=False)
    report = audit(train, test)
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
