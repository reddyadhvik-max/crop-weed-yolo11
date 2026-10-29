"""Crop/weed detection with YOLO11x. Run: .venv/bin/python train.py"""
import random, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
SRC = ROOT / "data/raw/agri_data/data"
DS = ROOT / "data/yolo"


def split():
    if (DS / "data.yaml").exists():
        return
    stems = sorted(p.stem for p in SRC.glob("*.jpeg"))
    random.Random(0).shuffle(stems)
    n = len(stems)
    parts = {"train": stems[: int(n * .8)], "val": stems[int(n * .8): int(n * .9)], "test": stems[int(n * .9):]}
    for name, items in parts.items():
        for sub in ("images", "labels"):
            (DS / sub / name).mkdir(parents=True, exist_ok=True)
        for s in items:
            shutil.copy(SRC / f"{s}.jpeg", DS / "images" / name / f"{s}.jpeg")
            shutil.copy(SRC / f"{s}.txt", DS / "labels" / name / f"{s}.txt")
    (DS / "data.yaml").write_text(
        f"path: {DS}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {{0: crop, 1: weed}}\n")
    assert sum(len(v) for v in parts.values()) == n


if __name__ == "__main__":
    split()
    last = ROOT / "runs/yolo11x/weights/last.pt"
    if last.exists():  # crashed/killed run -> pick up where it left off
        try:
            YOLO(last).train(resume=True)
        except AssertionError:  # "training is finished, nothing to resume" -> go straight to eval
            pass
    else:
        YOLO("yolo11x.pt").train(  # largest YOLO11 = max accuracy
            data=str(DS / "data.yaml"), epochs=300, patience=60, imgsz=640, batch=0.85,  # auto-batch to 85% of 8GB VRAM
            optimizer="AdamW", lr0=0.001, cos_lr=True, close_mosaic=20,
            degrees=10, flipud=0.5, mixup=0.1, copy_paste=0.1,  # top-down field shots: rotation/vertical flip are valid
            project=str(ROOT / "runs"), name="yolo11x", exist_ok=True, workers=8, seed=0, plots=True,
        )
    best = YOLO(ROOT / "runs/yolo11x/weights/best.pt")
    m = best.val(data=str(DS / "data.yaml"), split="test", imgsz=640, project=str(ROOT / "runs"), name="test", exist_ok=True)
    print(f"TEST mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} P={m.box.mp:.4f} R={m.box.mr:.4f}")
    best.predict(DS / "images/test", imgsz=640, save=True, project=str(ROOT / "runs"), name="predict", exist_ok=True)
