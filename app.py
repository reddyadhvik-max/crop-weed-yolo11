"""Crop/weed detector UI. Run: .venv/bin/python app.py  -> http://127.0.0.1:7860"""
from pathlib import Path
import cv2
import gradio as gr
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "runs/yolo11x/weights/best.pt"
model = YOLO(WEIGHTS)
COLORS = {"crop": (40, 200, 60), "weed": (230, 50, 50)}  # RGB


def detect(img, conf):
    """RGB numpy in -> (annotated RGB numpy, summary text, rows)."""
    if img is None:
        return None, "", []
    r = model.predict(img[..., ::-1], imgsz=640, conf=conf, verbose=False)[0]  # ultralytics expects BGR arrays
    names = [r.names[int(c)] for c in r.boxes.cls]
    out = img.copy()  # draw in RGB
    rows = []
    for (x1, y1, x2, y2), n, c in zip(r.boxes.xyxy.int().tolist(), names, r.boxes.conf.tolist()):
        col = COLORS[n]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        cv2.putText(out, f"{n} {c:.2f}", (x1 + 4, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        rows.append([n, round(c, 3), x1, y1, x2, y2])
    summary = f"🌱 crop: {names.count('crop')}   🌿 weed: {names.count('weed')}" if names else "No crop or weed detected"
    return out, summary, rows


with gr.Blocks(title="Crop & Weed Detector") as demo:
    gr.Markdown("# 🌱 Crop & Weed Detector\nYOLO11x trained on crop/weed field images. "
                "<span style='color:#28c83c'>■ crop</span> &nbsp; <span style='color:#e63232'>■ weed</span>")
    conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Confidence threshold")
    with gr.Tab("📷 Live Camera"):
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Camera feed")
            cam_out = gr.Image(label="Detections")
        cam_txt = gr.Markdown()
        cam.stream(lambda i, c: detect(i, c)[:2], [cam, conf], [cam_out, cam_txt], stream_every=0.15, concurrency_limit=1)
    with gr.Tab("🖼️ Upload Image"):
        with gr.Row():
            up = gr.Image(sources=["upload", "clipboard"], label="Your image")
            up_out = gr.Image(label="Detections")
        up_txt = gr.Markdown()
        table = gr.Dataframe(headers=["class", "confidence", "x1", "y1", "x2", "y2"], label="Boxes")
        btn = gr.Button("Detect", variant="primary")
        for ev in (btn.click, up.upload):
            ev(detect, [up, conf], [up_out, up_txt, table])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate"), inbrowser=True)
