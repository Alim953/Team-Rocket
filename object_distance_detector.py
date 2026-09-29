"""
Object Detection + Monocular Distance Estimation
=================================================

Detects objects (person, backpack, handbag, suitcase, and any other COCO
class you enable) with YOLOv8, then estimates the distance from the camera
to each object using the classic "triangle similarity" pinhole-camera model:

        distance = (real_world_height * focal_length) / pixel_height

This is the standard approach for single, uncalibrated-depth RGB cameras.
Read ACCURACY_NOTES.md (or the section at the bottom of README.md) before
trusting the numbers it prints -- monocular distance is an *estimate*, not
a lidar-grade measurement, and its accuracy depends entirely on:
    1) how well you calibrate FOCAL_LENGTH_PX for your specific camera, and
    2) how close the real object's size is to the KNOWN_HEIGHT_CM assumed
       for its class.

Usage
-----
    # Webcam, live, default classes (person, backpack, handbag, suitcase)
    python object_distance_detector.py --source 0

    # Video file, save annotated output
    python object_distance_detector.py --source input.mp4 --save

    # Single image
    python object_distance_detector.py --source photo.jpg --save

    # Higher-accuracy model (slower), custom confidence threshold
    python object_distance_detector.py --source 0 --model yolov8m.pt --conf 0.5

    # Use your own calibrated focal length (see calibrate_focal_length.py)
    python object_distance_detector.py --source 0 --focal-length 812.5

    # Track objects across frames (stable IDs, smoother boxes) instead of
    # per-frame detection
    python object_distance_detector.py --source 0 --track
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Reference real-world sizes (cm) used for distance estimation.
# These are population-average figures. Override them with --known-sizes
# a_json_file.json if you know the actual size of the object you care about
# (e.g. you always carry the same backpack -- measure it once, use that).
# Values are the object's *height* in cm as it typically appears upright
# in a bounding box, since height is far less affected by viewing angle
# / pose than width.
# --------------------------------------------------------------------------
DEFAULT_KNOWN_HEIGHT_CM = {
    "person": 170.0,      # average adult standing height
    "backpack": 45.0,
    "handbag": 28.0,
    "suitcase": 65.0,
    "bottle": 24.0,
    "cell phone": 14.0,
    "laptop": 25.0,
    "chair": 90.0,
    "car": 150.0,
    "bicycle": 105.0,
    "motorcycle": 115.0,
    "dog": 55.0,
    "cat": 30.0,
}

# Default classes we actually keep in the output ("person" + common "bag"
# categories in COCO -- COCO has no single generic "bag" class, so we
# include the three closest ones).
DEFAULT_CLASSES = ["person", "backpack", "handbag", "suitcase"]

# A reasonable starting focal length (pixels) for a 720p-ish webcam.
# THIS IS A GUESS. Run calibrate_focal_length.py with your own camera and
# pass the result via --focal-length for real accuracy.
DEFAULT_FOCAL_LENGTH_PX = 700.0


def load_known_heights(path: str | None) -> dict:
    heights = dict(DEFAULT_KNOWN_HEIGHT_CM)
    if path:
        with open(path, "r") as f:
            heights.update(json.load(f))
    return heights


def estimate_distance_cm(known_height_cm: float, focal_length_px: float, pixel_height: float) -> float | None:
    """Pinhole camera / triangle-similarity distance estimate."""
    if pixel_height <= 0:
        return None
    return (known_height_cm * focal_length_px) / pixel_height


def color_for_class(name: str) -> tuple:
    # Stable, distinct colors per class (BGR)
    palette = {
        "person": (60, 220, 60),
        "backpack": (60, 160, 255),
        "handbag": (255, 120, 60),
        "suitcase": (60, 60, 255),
    }
    if name in palette:
        return palette[name]
    # deterministic pseudo-random color for any other class
    h = abs(hash(name)) % (256 ** 3)
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)


def draw_detection(frame, box_xyxy, label, distance_cm, conf, color, track_id=None):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if distance_cm is not None:
        if distance_cm >= 100:
            dist_txt = f"{distance_cm / 100:.2f} m"
        else:
            dist_txt = f"{distance_cm:.0f} cm"
    else:
        dist_txt = "n/a"

    id_txt = f" #{track_id}" if track_id is not None else ""
    text = f"{label}{id_txt} {conf * 100:.0f}%  |  {dist_txt}"

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)


def run(args):
    known_heights = load_known_heights(args.known_sizes)
    wanted_classes = set(c.strip().lower() for c in args.classes.split(",")) if args.classes else None

    model = YOLO(args.model)
    class_names = model.names  # dict: id -> name

    # Resolve which class ids we keep
    keep_ids = None
    if wanted_classes:
        keep_ids = [cid for cid, name in class_names.items() if name.lower() in wanted_classes]
        if not keep_ids:
            print(f"[warn] none of the requested classes {wanted_classes} exist in this model's "
                  f"class list; detecting ALL classes instead.")
            keep_ids = None

    # Decide source type
    source = args.source
    is_webcam = source.isdigit()
    src_for_cv = int(source) if is_webcam else source

    is_image = (not is_webcam) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    writer = None
    out_path = None

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[error] could not read image: {source}")
            sys.exit(1)
        frames_iter = [frame]
    else:
        cap = cv2.VideoCapture(src_for_cv)
        if not cap.isOpened():
            print(f"[error] could not open video source: {source}")
            sys.exit(1)
        if args.save:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = args.output or "output_annotated.mp4"
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_t = time.time()

    def process_frame(frame):
        if args.track:
            results = model.track(frame, persist=True, conf=args.conf, iou=args.iou,
                                   classes=keep_ids, verbose=False)
        else:
            results = model.predict(frame, conf=args.conf, iou=args.iou,
                                     classes=keep_ids, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return frame

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if (args.track and r.boxes.id is not None) else [None] * len(cls_ids)

        for box, conf, cid, tid in zip(boxes_xyxy, confs, cls_ids, ids):
            name = class_names.get(cid, str(cid))
            x1, y1, x2, y2 = box
            pixel_height = y2 - y1

            known_h = known_heights.get(name)
            distance_cm = estimate_distance_cm(known_h, args.focal_length, pixel_height) if known_h else None

            draw_detection(frame, box, name, distance_cm, conf, color_for_class(name), tid)

        return frame

    if is_image:
        frame = process_frame(frames_iter[0])
        if args.save:
            out_path = args.output or "output_annotated.jpg"
            cv2.imwrite(out_path, frame)
            print(f"[ok] saved annotated image to {out_path}")
        if args.show:
            cv2.imshow("detections", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(frame)

        frame_count += 1
        now = time.time()
        fps_disp = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[ok] saved annotated video to {out_path}")
    if args.show:
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 object detection with monocular distance estimation")
    p.add_argument("--source", default="0",
                    help="Webcam index (e.g. 0), or path to an image/video file")
    p.add_argument("--model", default="yolov8m.pt",
                    help="YOLOv8 weights: yolov8n.pt (fastest) ... yolov8x.pt (most accurate)")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="Comma-separated class names to keep, e.g. 'person,backpack'. "
                         "Use '' to detect every COCO class.")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    p.add_argument("--focal-length", type=float, default=DEFAULT_FOCAL_LENGTH_PX,
                    help="Calibrated focal length in pixels (see calibrate_focal_length.py)")
    p.add_argument("--known-sizes", default=None,
                    help="Path to a JSON file overriding real-world object heights in cm, "
                         "e.g. {\"person\": 178}")
    p.add_argument("--track", action="store_true",
                    help="Use YOLO's built-in tracker for stable IDs across frames (video/webcam only)")
    p.add_argument("--save", action="store_true", help="Save annotated output to disk")
    p.add_argument("--output", default=None, help="Output file path (defaults auto-named)")
    p.add_argument("--show", action="store_true", default=True,
                    help="Show a live preview window (ignored in headless environments)")
    p.add_argument("--no-show", dest="show", action="store_false",
                    help="Disable the preview window (use for headless/server runs)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
