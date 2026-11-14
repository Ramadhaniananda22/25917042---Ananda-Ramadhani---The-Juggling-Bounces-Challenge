"""Sanity test of the detect + evaluate stages on a synthetic 3-ball juggling trajectory."""
import numpy as np
import bounce_counter as bc

fps, FLOOR, HAND = 30, 900.0, 600.0
rng = np.random.RandomState(0)
tracks, gt = {}, []

# Each ball: arc from hand (y=600) down to floor (y=900) [bounce], back up to hand [catch],
# then thrown up (peak at y=300) and falls to hand again.
for tid, offset in enumerate([0, 10, 20], start=1):
    rows, f = [], offset
    for cycle in range(6):
        # fall hand -> floor (12 frames), bounce, rise floor -> hand (12 frames)
        for k in range(12):
            rows.append((f, 400 + 80 * tid, HAND + (FLOOR - HAND) * (k / 11) ** 2, 40, 40)); f += 1
        gt.append(f / fps)
        for k in range(12):
            rows.append((f, 400 + 80 * tid, FLOOR - (FLOOR - HAND) * (k / 11) ** 0.5, 40, 40)); f += 1
        # toss up and catch (no floor contact): hand-height local MAX that must be rejected
        for k in range(10):
            rows.append((f, 400 + 80 * tid, HAND - 250 * np.sin(np.pi * k / 9), 40, 40)); f += 1
    arr = np.array(rows, float)
    arr[:, 2] += rng.normal(0, 2, len(arr))
    # drop 8% of detections (simulated misses)
    keep = rng.rand(len(arr)) > 0.08
    tracks[tid] = arr[keep]

events, floor = bc.detect_bounces(tracks, fps)
print("floor estimate:", floor, "(true", FLOOR, ")")
bc.report([e["time"] for e in events], gt_count=len(gt), gt_times=sorted(gt), tol=0.2)
