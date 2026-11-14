"""
Optional: fine-tune YOLO on basketballs from this video (or similar footage).

1. Extract frames:      python finetune.py extract --video juggling.mp4 --out frames --every 5
2. Label them (Roboflow / CVAT / LabelImg), single class "basketball", YOLO format.
3. Create data.yaml:
       path: dataset
       train: images/train
       val: images/val
       names: {0: basketball}
4. Train:               python finetune.py train --data data.yaml

Note: to avoid leaking validation data, split by TIME (e.g. first 80% of the video for
train, last 20% for val), not by random frame.
"""
import argparse
import os


def extract(video, out, every):
    import cv2
    os.makedirs(out, exist_ok=True)
    cap, i, n = cv2.VideoCapture(video), 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            cv2.imwrite(os.path.join(out, f"f{i:06d}.jpg"), fr)
            n += 1
        i += 1
    print(f"saved {n} frames to {out}")


def train(data, weights, epochs, imgsz):
    from ultralytics import YOLO
    model = YOLO(weights)
    model.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=8,
        # --- augmentation aimed at edge cases -------------------------------
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # lighting / colour variation
        degrees=10, translate=0.1, scale=0.5,  # small rotations, shifts, scale
        fliplr=0.5,                           # horizontal flip
        mosaic=1.0, mixup=0.1,                # context variety, partial occlusion
        # Motion blur: Ultralytics applies Albumentations (Blur/MedianBlur) automatically
        # if `pip install albumentations` is present. For stronger, directional motion
        # blur, pre-process the training images with albumentations.MotionBlur
        # (blur_limit=(7, 21), p=0.5) before training.
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--video", required=True)
    e.add_argument("--out", default="frames")
    e.add_argument("--every", type=int, default=5)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--weights", default="yolov8m.pt")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.video, a.out, a.every)
    else:
        train(a.data, a.weights, a.epochs, a.imgsz)
