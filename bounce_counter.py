"""
Juggling Bounces Challenge - bounce counting pipeline.

Stages
------
1. track    : YOLO detector + ByteTrack -> tracks.csv (frame, id, cx, cy, w, h, conf)
2. detect   : trajectory analysis (peak detection on vertical position) -> bounces.csv
3. evaluate : compare against manual ground truth (count + timestamps) -> TP/FP/FN
4. render   : annotated demo video (boxes, trails, IDs, live bounce counter)

Image coordinates: y grows DOWNWARD, so a bounce (ball touching the floor) is a
LOCAL MAXIMUM of cy. Hand catches are also local maxima of cy but happen higher
in the frame (hip height), so they are removed by a floor-line filter.

Usage
-----
python bounce_counter.py run --video juggling.mp4 --out outputs --gt 25
python bounce_counter.py track --video juggling.mp4 --out outputs
python bounce_counter.py detect --tracks outputs/tracks.csv --fps 30 --out outputs
python bounce_counter.py evaluate --bounces outputs/bounces.csv --gt-times gt_times.txt
python bounce_counter.py render --video juggling.mp4 --tracks outputs/tracks.csv \
       --bounces outputs/bounces.csv --out outputs/demo.mp4
"""
import argparse
import csv
import os
from collections import defaultdict, deque

import numpy as np

SPORTS_BALL_COCO_ID = 32  # COCO class "sports ball"


# --------------------------------------------------------------------------- #
# Stage 1: detection + tracking
# --------------------------------------------------------------------------- #
def run_tracking(video, out_dir, weights="yolov8m.pt", conf=0.15, imgsz=1280,
                 tracker="bytetrack.yaml", classes=(SPORTS_BALL_COCO_ID,)):
    """Run YOLO + ByteTrack and save per-frame detections to tracks.csv."""
    from ultralytics import YOLO  # imported here so other stages need no GPU stack

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)
    rows, fps = [], None

    # Low conf threshold: motion-blurred balls score low; filtering happens later.
    results = model.track(source=video, stream=True, persist=True, conf=conf,
                          imgsz=imgsz, tracker=tracker, classes=list(classes),
                          verbose=False)
    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().cpu().tolist()
        xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for tid, (cx, cy, w, h), c in zip(ids, xywh, confs):
            rows.append((frame_idx, tid, float(cx), float(cy), float(w), float(h), float(c)))

    path = os.path.join(out_dir, "tracks.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "id", "cx", "cy", "w", "h", "conf"])
        wr.writerows(rows)
    print(f"[track] {len(rows)} detections -> {path}")
    return path


# --------------------------------------------------------------------------- #
# Stage 2: trajectory analysis -> bounce events
# --------------------------------------------------------------------------- #
def load_tracks(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            tracks[int(r["id"])].append((int(r["frame"]), float(r["cx"]), float(r["cy"]),
                                         float(r["w"]), float(r["h"])))
    return {k: np.array(sorted(v)) for k, v in tracks.items()}


def interpolate_gaps(arr, max_gap):
    """Linearly fill short gaps (missed detections, e.g. motion blur/occlusion)."""
    frames = arr[:, 0].astype(int)
    full = np.arange(frames[0], frames[-1] + 1)
    out = np.column_stack([np.interp(full, frames, arr[:, c]) for c in range(1, arr.shape[1])])
    gap_ok = np.ones(len(full), bool)
    # mark frames inside gaps longer than max_gap as invalid
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a - 1 > max_gap:
            gap_ok[(full > a) & (full < b)] = False
    return full[gap_ok], out[gap_ok]


def two_means_1d(x, iters=50):
    """Tiny 1-D 2-means. Returns (low_center, high_center)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    for _ in range(iters):
        mid = (lo + hi) / 2
        a, b = x[x <= mid], x[x > mid]
        if len(a) == 0 or len(b) == 0:
            break
        lo, hi = a.mean(), b.mean()
    return lo, hi


def detect_bounces(tracks, fps, prominence_ratio=1.5, min_track_len=8, max_gap=6,
                   smooth_window=5, floor_y=None, floor_tol_ratio=1.5,
                   dedup_sec=0.12, dedup_px_ratio=3.0):
    """
    Returns list of dicts: {frame, time, track_id, x, y}.

    Steps per track
    ---------------
    a. interpolate short gaps
    b. smooth cy (moving average) to suppress detector jitter
    c. find local maxima of cy (lowest point on screen) with scipy.signal.find_peaks,
       requiring prominence >= prominence_ratio * ball height (ignores tiny wobbles)
    Global steps
    ------------
    d. estimate floor line (2-means over candidate peak heights, take the LOWER
       cluster on screen = larger y) unless --floor-y is given; keep only peaks
       within floor_tol_ratio * ball height of that line  -> removes hand catches
    e. de-duplicate events produced by tracker ID switches (same place, same time)
    """
    from scipy.signal import find_peaks

    cands = []
    for tid, arr in tracks.items():
        if len(arr) < min_track_len:
            continue
        frames, vals = interpolate_gaps(arr, max_gap)
        if len(frames) < min_track_len:
            continue
        cx, cy, w, h = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
        if smooth_window > 1 and len(cy) >= smooth_window:
            k = np.ones(smooth_window) / smooth_window
            cy_s = np.convolve(np.pad(cy, smooth_window // 2, mode="edge"), k, mode="valid")
        else:
            cy_s = cy
        ball_h = float(np.median(h))
        peaks, _ = find_peaks(cy_s, prominence=prominence_ratio * ball_h)
        for p in peaks:
            cands.append(dict(frame=int(frames[p]), track_id=int(tid), x=float(cx[p]),
                              y=float(cy_s[p]), ball_h=ball_h))
    if not cands:
        return [], None

    ys = np.array([c["y"] for c in cands])
    if floor_y is None:
        lo, hi = two_means_1d(ys)
        # Only trust the 2-cluster split if clusters are clearly separated
        median_h = float(np.median([c["ball_h"] for c in cands]))
        floor_y = hi if (hi - lo) > 2 * median_h else float(np.max(ys))
    kept = [c for c in cands if c["y"] >= floor_y - floor_tol_ratio * c["ball_h"]]

    kept.sort(key=lambda c: c["frame"])
    final = []
    for c in kept:
        dup = False
        for k in final[::-1]:
            if (c["frame"] - k["frame"]) / fps > dedup_sec:
                break
            if abs(c["x"] - k["x"]) < dedup_px_ratio * c["ball_h"]:
                dup = True
                break
        if not dup:
            final.append(c)
    for c in final:
        c["time"] = c["frame"] / fps
    return final, floor_y


def save_bounces(events, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bounces.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bounce_no", "frame", "time_sec", "track_id", "x", "y"])
        for i, e in enumerate(events, 1):
            wr.writerow([i, e["frame"], f"{e['time']:.3f}", e["track_id"],
                         f"{e['x']:.1f}", f"{e['y']:.1f}"])
    return path


# --------------------------------------------------------------------------- #
# Stage 3: evaluation against manual ground truth
# --------------------------------------------------------------------------- #
def load_times(path):
    with open(path) as f:
        return sorted(float(l.strip()) for l in f if l.strip() and not l.startswith("#"))


def match_events(pred_t, gt_t, tol):
    """Greedy one-to-one matching of predicted vs ground-truth timestamps."""
    used, tp, fp = set(), [], []
    for p in pred_t:
        best, bd = None, tol
        for i, g in enumerate(gt_t):
            if i not in used and abs(p - g) <= bd:
                best, bd = i, abs(p - g)
        if best is None:
            fp.append(p)
        else:
            used.add(best)
            tp.append((p, gt_t[best]))
    fn = [g for i, g in enumerate(gt_t) if i not in used]
    return tp, fp, fn


def report(pred_times, gt_count=None, gt_times=None, tol=0.15):
    n = len(pred_times)
    print(f"\nPredicted bounces: {n}")
    if gt_count is None and gt_times is not None:
        gt_count = len(gt_times)
    if gt_count is not None:
        err = n - gt_count
        ape = abs(err) / gt_count * 100
        print(f"Ground truth     : {gt_count}")
        kind = "exact" if err == 0 else ("over-count" if err > 0 else "under-count")
        print(f"Absolute error   : {abs(err)}  ({kind})")
        print(f"APE              : {ape:.2f}%   accuracy ~ {100 - ape:.2f}%")
    if gt_times:
        tp, fp, fn = match_events(pred_times, gt_times, tol)
        prec = len(tp) / max(len(tp) + len(fp), 1)
        rec = len(tp) / max(len(tp) + len(fn), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"\nEvent-level (tolerance +/-{tol}s): TP={len(tp)} FP={len(fp)} FN={len(fn)}")
        print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print("\nOVER-COUNTED (false positives) at t =", [f"{t:.2f}s" for t in fp])
        print("MISSED (false negatives) at t     =", [f"{t:.2f}s" for t in fn])
        print("-> Inspect these timestamps in the video for the Failure Analysis section.")


# --------------------------------------------------------------------------- #
# Stage 4: demo video
# --------------------------------------------------------------------------- #
def render_video(video, tracks_csv, bounces_csv, out_path, trail=30, flash=12):
    import cv2

    per_frame = defaultdict(list)
    with open(tracks_csv) as f:
        for r in csv.DictReader(f):
            per_frame[int(r["frame"])].append((int(r["id"]), float(r["cx"]), float(r["cy"]),
                                               float(r["w"]), float(r["h"]), float(r["conf"])))
    events = []
    with open(bounces_csv) as f:
        for r in csv.DictReader(f):
            events.append((int(r["frame"]), float(r["x"]), float(r["y"])))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = defaultdict(lambda: deque(maxlen=trail))
    rng = np.random.RandomState(1)
    colors = {}

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, cx, cy, w, h, c in per_frame.get(i, []):
            col = colors.setdefault(tid, tuple(int(v) for v in rng.randint(60, 255, 3)))
            trails[tid].append((int(cx), int(cy)))
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"ID {tid} {c:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            pts = list(trails[tid])
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(frame, a, b, col, 2)
        count = sum(1 for e in events if e[0] <= i)
        for fe, ex, ey in events:
            if 0 <= i - fe < flash:
                cv2.circle(frame, (int(ex), int(ey)), 30, (0, 0, 255), 3)
                cv2.putText(frame, "BOUNCE!", (int(ex) - 40, int(ey) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 10), (270, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Bounces: {count}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()
    print(f"[render] saved {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_detect_args(p):
    p.add_argument("--prominence-ratio", type=float, default=1.5)
    p.add_argument("--floor-y", type=float, default=None, help="manual floor line (pixels)")
    p.add_argument("--floor-tol-ratio", type=float, default=1.5)
    p.add_argument("--max-gap", type=int, default=6)
    p.add_argument("--smooth", type=int, default=5)


def do_detect(tracks_csv, fps, out_dir, a):
    events, floor_y = detect_bounces(
        load_tracks(tracks_csv), fps, prominence_ratio=a.prominence_ratio,
        floor_y=a.floor_y, floor_tol_ratio=a.floor_tol_ratio,
        max_gap=a.max_gap, smooth_window=a.smooth)
    path = save_bounces(events, out_dir)
    print(f"[detect] floor_y={floor_y}  bounces={len(events)} -> {path}")
    return events


def video_fps(video):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("track")
    t.add_argument("--video", required=True)
    t.add_argument("--out", default="outputs")
    t.add_argument("--weights", default="yolov8m.pt")
    t.add_argument("--conf", type=float, default=0.15)
    t.add_argument("--imgsz", type=int, default=1280)

    d = sub.add_parser("detect")
    d.add_argument("--tracks", required=True)
    d.add_argument("--fps", type=float, required=True)
    d.add_argument("--out", default="outputs")
    add_detect_args(d)

    e = sub.add_parser("evaluate")
    e.add_argument("--bounces", required=True)
    e.add_argument("--gt", type=int, help="ground-truth bounce count")
    e.add_argument("--gt-times", help="text file: one ground-truth bounce time (sec) per line")
    e.add_argument("--tol", type=float, default=0.15)

    r = sub.add_parser("render")
    r.add_argument("--video", required=True)
    r.add_argument("--tracks", required=True)
    r.add_argument("--bounces", required=True)
    r.add_argument("--out", default="outputs/demo.mp4")

    u = sub.add_parser("run", help="track + detect + evaluate + render")
    u.add_argument("--video", required=True)
    u.add_argument("--out", default="outputs")
    u.add_argument("--weights", default="yolov8m.pt")
    u.add_argument("--conf", type=float, default=0.15)
    u.add_argument("--imgsz", type=int, default=1280)
    u.add_argument("--gt", type=int)
    u.add_argument("--gt-times")
    u.add_argument("--tol", type=float, default=0.15)
    add_detect_args(u)

    a = ap.parse_args()
    if a.cmd == "track":
        run_tracking(a.video, a.out, a.weights, a.conf, a.imgsz)
    elif a.cmd == "detect":
        do_detect(a.tracks, a.fps, a.out, a)
    elif a.cmd == "evaluate":
        times = []
        with open(a.bounces) as f:
            times = [float(r["time_sec"]) for r in csv.DictReader(f)]
        report(times, a.gt, load_times(a.gt_times) if a.gt_times else None, a.tol)
    elif a.cmd == "render":
        render_video(a.video, a.tracks, a.bounces, a.out)
    elif a.cmd == "run":
        tr = run_tracking(a.video, a.out, a.weights, a.conf, a.imgsz)
        ev = do_detect(tr, video_fps(a.video), a.out, a)
        report([e["time"] for e in ev], a.gt, load_times(a.gt_times) if a.gt_times else None, a.tol)
        render_video(a.video, tr, os.path.join(a.out, "bounces.csv"),
                     os.path.join(a.out, "demo.mp4"))


if __name__ == "__main__":
    main()
