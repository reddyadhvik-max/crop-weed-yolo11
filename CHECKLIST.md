# Project Checklist — Crop & Weed Detection (YOLO11x)

## ✅ Data
- [x] Source: Kaggle "Crop and Weed Detection Data with Bounding Boxes" (ravirajsinh45)
- [x] 1300 images, all 512×512 JPEG, every image labelled (YOLO txt format)
- [x] 2 classes: `0 = crop` (1212 boxes), `1 = weed` (860 boxes)
- [x] Split 80/10/10 with fixed seed 0 → train 1040 / val 130 / test 130
- [x] Test split never used in training or model selection

## ✅ Model & training
- [x] Model: YOLO11x (largest YOLO11), 56.9 M parameters, 357 layers
- [x] Compute: 195.8 GFLOPs per image at 640×640
- [x] Started from COCO-pretrained weights (transfer learning)
- [x] Input size 640, AdamW optimizer, lr 0.001, cosine LR schedule
- [x] Augmentations: mosaic, mixup 0.1, copy-paste 0.1, rotation ±10°, horizontal + vertical flips, HSV colour shifts
- [x] Batch size 5 (auto-sized to fit 8 GB VRAM), mixed precision (AMP)
- [x] Up to 300 epochs, early stopping patience 60 → stopped at ~epoch 229 (best epoch 169)
- [x] Total training time ≈ 3 h on RTX 5050 Laptop GPU
- [x] Auto-resume supervisor (`supervise.sh`) so crashes restart from last checkpoint — 0 restarts needed

## ✅ Results (held-out test set, 130 images)
| Metric | Score |
|---|---|
| mAP50 | 0.896 |
| mAP50-95 | 0.595 |
| Precision / Recall | 0.856 / 0.831 |
| AP50 crop / weed | 0.866 / 0.926 |
| Image-level class match | 128/130 (98.5%) |

- [x] Accuracy test suite (`test_model.py`) — ALL PASSED
  - mAP50 ≥ 0.85 ✅ · every class AP50 ≥ 0.80 ✅ · image-level match ≥ 90% ✅ · UI pipeline returns boxes ✅

### What the metrics mean
- **IoU (Intersection over Union)**: how much a predicted box overlaps the true box (0 = no overlap, 1 = perfect). A detection counts as "correct" only if IoU is above a threshold.
- **Precision (0.856)**: of everything the model flagged, 85.6% was right. Low precision = false alarms (e.g. calling soil a weed).
- **Recall (0.831)**: of all real plants in the images, the model found 83.1%. Low recall = missed plants.
- **AP (Average Precision)**: area under the precision-recall curve for one class, sweeping the confidence threshold — a single score combining both precision and recall.
- **AP50 crop / weed (0.866 / 0.926)**: AP per class, with a box counting as correct at IoU ≥ 0.5. Weeds are detected slightly better than crops.
- **mAP50 (0.896)**: mean of the per-class AP50s — the headline "is it finding the right things in roughly the right place" score.
- **mAP50-95 (0.595)**: AP averaged over strict IoU thresholds 0.50, 0.55 … 0.95. Rewards pixel-tight boxes; always much lower than mAP50. It's lower here mostly because plant edges are fuzzy and annotators drew loose boxes.
- **Image-level class match (128/130)**: for each test image, did the model report exactly the same set of classes (crop / weed / both) as the ground truth? This is the "does it tell crop from weed" answer in plain terms.

## ✅ Compute & hardware
- [x] Model cost: **195.8 GFLOPs per frame** (≈ 98 billion multiply-adds) at 640×640
- [x] Your laptop:
  - GPU: NVIDIA RTX 5050 Laptop (Blackwell, 8 GB) — **~440 AI TOPS** marketing figure (low-precision INT8/FP4 Tensor-core peak)
  - CPU: AMD Ryzen 7 250 (8C/16T, Zen 4) with Ryzen AI NPU **16 TOPS**, platform total **38 TOPS**
- [x] Measured speed on this laptop (640×640, single image):

| Device | Time / image | FPS | Effective compute used |
|---|---|---|---|
| RTX 5050, FP16 | 15.6 ms | **64 FPS** | ≈ 12.5 TOPS |
| RTX 5050, FP32 (UI default) | 27.1 ms | 37 FPS | ≈ 7.2 TOPS |
| Ryzen 7 250 CPU | 370 ms | 2.7 FPS | ≈ 0.5 TOPS |

- [x] Compute needed for smooth real-time:
  - 30 FPS × 195.8 GFLOPs ≈ **5.9 TOPS of actual work**
  - Real chips reach only ~10–30% of their advertised TOPS on a model like this, so budget **≈ 20–60 advertised TOPS** for 30 FPS with YOLO11x
  - Your RTX 5050 clears this easily (37–64 FPS measured); CPU-only does not (2.7 FPS)
  - Edge-device option: switch to YOLO11n/s (6.5 / 21.5 GFLOPs) for Jetson Orin Nano / Raspberry Pi + accelerator

## ✅ App / UI (`app.py`, Gradio, green theme)
- [x] Tab 1 📷 Live Camera — browser webcam stream, crop = green boxes, weed = red boxes, live counts
- [x] Tab 2 🖼️ Upload Image — pick image from laptop, original + detections side by side, table of boxes with confidence
- [x] Shared confidence slider (default 0.35)
- [x] Upload tab verified in browser with a held-out test image
- [ ] Live camera — needs browser camera permission, test by hand

## ✅ Deliverables
- [x] Code + weights + results: `~/Desktop/crop_weed_yolo.zip` (127 MB)
- [x] Public repo: https://github.com/reddyadhvik-max/crop-weed-yolo11
- [x] Weights download (114 MB, over GitHub's 100 MB file limit): release v1.0
- [x] `RESULTS.txt`, 130 annotated test predictions in `runs/predict/`, training curves in `runs/yolo11x/`

## Known limits
- Trained on top-down field photos of young plants; side angles, mature plants, or very different soil/lighting will lower accuracy.
- Only 1300 images — more varied data is the biggest lever for further gains.
