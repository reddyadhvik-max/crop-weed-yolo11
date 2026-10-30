# 🌱 Crop & Weed Detection — YOLO11x

YOLO11x trained on the Kaggle [Crop and Weed Detection](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) dataset (1300 images, 512×512, classes: crop, weed), with a Gradio UI for live webcam and image-upload detection.

## Results (held-out test split, 130 images never seen in training)

| Metric | Score |
|---|---|
| mAP50 | **0.896** |
| mAP50-95 | 0.595 |
| Precision / Recall | 0.856 / 0.831 |
| AP50 crop / weed | 0.866 / 0.926 |
| Image-level class match | 128/130 (98.5%) |

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv/bin/python ultralytics gradio
```

Download `best.pt` from the [latest release](../../releases/latest) into `runs/yolo11x/weights/best.pt`.

## Use

```bash
.venv/bin/python app.py         # UI: 📷 Live Camera tab + 🖼️ Upload Image tab
.venv/bin/python test_model.py  # accuracy tests (needs dataset split in data/yolo)
```

## Train from scratch

Unzip the Kaggle dataset into `data/raw/`, then:

```bash
./supervise.sh   # runs train.py (80/10/10 split, YOLO11x, 300 epochs, early stop), auto-resumes on crash
```
