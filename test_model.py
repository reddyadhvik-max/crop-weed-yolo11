"""Accuracy + sanity checks on the held-out test split. Run: .venv/bin/python test_model.py"""
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent
DS = ROOT / "data/yolo"
model = YOLO(ROOT / "runs/yolo11x/weights/best.pt")

# 1. Standard COCO-style metrics on unseen test images
m = model.val(data=str(DS / "data.yaml"), split="test", imgsz=640, verbose=False, plots=False)
ap50 = dict(zip(m.names.values(), m.box.ap50))
print(f"test mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} P={m.box.mp:.3f} R={m.box.mr:.3f} per-class AP50={ap50}")

# 2. Per-image: does the model say "crop" / "weed" where the ground truth does?
imgs = sorted((DS / "images/test").glob("*.jpeg"))
hits = 0
for p in imgs:
    gt = {int(l.split()[0]) for l in (DS / "labels/test" / f"{p.stem}.txt").read_text().split("\n") if l.strip()}
    pred = {int(c) for c in model.predict(p, imgsz=640, conf=0.35, verbose=False)[0].boxes.cls}
    hits += gt == pred
img_acc = hits / len(imgs)
print(f"image-level class match: {hits}/{len(imgs)} = {img_acc:.3f}")

# 3. The UI's detect() path returns boxes on a real test image
import app
out, summary, rows = app.detect(np.array(Image.open(imgs[0]).convert("RGB")), 0.35)
assert out.shape == (512, 512, 3) and rows, summary
print("UI detect():", summary)

assert m.box.map50 >= 0.85, f"mAP50 {m.box.map50:.3f} < 0.85"
assert min(ap50.values()) >= 0.80, f"weakest class AP50 {ap50}"
assert img_acc >= 0.90, f"image-level accuracy {img_acc:.3f} < 0.90"
print("ALL TESTS PASSED")
