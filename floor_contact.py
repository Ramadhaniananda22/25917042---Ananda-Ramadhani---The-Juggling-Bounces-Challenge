"""
Hitung pantulan dari deteksi bola di dekat lantai, tanpa bergantung pada ID tracker.

Alasan: pada video ini ByteTrack memecah lintasan menjadi ratusan ID pendek, sehingga
analisis puncak lintasan per ID tidak menghasilkan pantulan. Metode ini hanya memakai
posisi deteksi (tracks.csv dari bounce_counter.py track).

Langkah:
1. Garis lantai = persentil ke-95 dari posisi vertikal (cy) semua deteksi, atau --floor-y manual.
2. Ambil deteksi yang cy-nya berada dalam jarak (tol x tinggi bola) dari garis lantai.
3. Kelompokkan deteksi yang berdekatan dalam waktu (<= max_gap frame) dan posisi
   horizontal (<= max_dx x tinggi bola). Satu kelompok = satu pantulan.

Pemakaian:
    python floor_contact.py out1 out2 out3 --fps 30 --durasi-menit 20
"""
import argparse

import numpy as np
import pandas as pd


def hitung(folder, fps=30, floor_y=None, tol=1.0, max_gap=3, max_dx=2.0, persen=95):
    t = pd.read_csv(f"{folder}/tracks.csv").sort_values("frame").reset_index(drop=True)
    h = float(t.h.median())
    if floor_y is None:
        floor_y = float(np.percentile(t.cy, persen))
    near = t[t.cy >= floor_y - tol * h]
    clusters = []
    for r in near.itertuples():
        placed = False
        for c in clusters[::-1]:
            if r.frame - c["last"] > max_gap:
                break
            if abs(r.cx - c["x"]) <= max_dx * h:
                c["last"] = r.frame
                c["x"] = r.cx
                c["pts"].append((r.cy, r.frame, r.cx))
                placed = True
                break
        if not placed:
            clusters.append(dict(last=r.frame, x=r.cx, pts=[(r.cy, r.frame, r.cx)]))
    clusters.sort(key=lambda c: min(p[1] for p in c["pts"]))
    rows = []
    for i, c in enumerate(clusters, 1):
        cy, fr, cx = max(c["pts"])
        rows.append((i, fr, round(fr / fps, 3), 0, round(cx, 1), round(cy, 1)))
    pd.DataFrame(rows, columns=["bounce_no", "frame", "time_sec", "track_id", "x", "y"]).to_csv(
        f"{folder}/bounces.csv", index=False)
    print(f"{folder}: lantai={floor_y:.0f}px, tinggi bola={h:.0f}px, pantulan={len(rows)}")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--floor-y", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--max-dx", type=float, default=2.0)
    ap.add_argument("--durasi-menit", type=float, default=None)
    a = ap.parse_args()
    n = [hitung(f, a.fps, a.floor_y, a.tol, a.max_gap, a.max_dx) for f in a.folders]
    print(f"Rata-rata: {sum(n) / len(n):.0f} pantulan per klip")
    if a.durasi_menit:
        print(f"Perkiraan {a.durasi_menit:.0f} menit: kurang lebih "
              f"{sum(n) / len(n) * a.durasi_menit:.0f} pantulan (jika tiap klip 1 menit)")
