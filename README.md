# Juggling Bounces Challenge - Penghitung Pantulan Bola

**Video demonstrasi:** https://drive.google.com/drive/folders/1tMUhr1SAaDo3lhlAiAERCY_fon4yLH4U?usp=sharing

Menghitung pantulan bola basket pada video juggling dengan YOLOv8 (deteksi), ByteTrack
(tracking), dan penghitungan kontak lantai dari posisi deteksi.

## Alur

```
video -> YOLOv8 (kelas sports ball) -> ByteTrack -> tracks.csv
      -> deteksi dekat garis lantai -> pengelompokan waktu + posisi -> bounces.csv
      -> video demo (kotak, ID, jejak, penanda BOUNCE!, penghitung berjalan)
```

Catatan: pada video ini ByteTrack memecah lintasan menjadi ratusan ID pendek, sehingga
metode puncak lintasan per ID (`bounce_counter.py detect`) tidak menemukan pantulan.
Penghitungan yang dipakai untuk hasil akhir adalah `floor_contact.py`, yang tidak
bergantung pada ID tracker.

## File

- `bounce_counter.py` : tracking (`run`/`detect`/`render`), termasuk deteksi puncak lintasan
- `floor_contact.py`  : penghitungan pantulan dari kontak lantai (dipakai untuk hasil akhir)
- `finetune.py`       : ekstraksi frame dan skrip fine-tuning (opsional, tidak dijalankan di proyek ini)
- `synthetic_test.py` : uji logika pada lintasan sintetis
- `25917042 - Ananda Ramadhani - The Juggling Bounces Challenge.ipynb` : notebook Colab, jalankan dari atas ke bawah

## Cara menjalankan (Google Colab, GPU T4)

```bash
pip install ultralytics lap yt-dlp scipy albumentations opencv-python
yt-dlp -f "bv*[height<=1080]+ba/b" --merge-output-format mp4 -o juggling.mp4 "https://www.youtube.com/watch?v=oWqwY1MLBck"

# potong klip 1 menit
ffmpeg -ss 60 -t 60 -i juggling.mp4 -an seg1.mp4

# deteksi + tracking
python bounce_counter.py track --video seg1.mp4 --out out1 --imgsz 960 --weights yolov8s.pt

# hitung pantulan
python floor_contact.py out1 --fps 30

# video demo
python bounce_counter.py render --video seg1.mp4 --tracks out1/tracks.csv --bounces out1/bounces.csv --out out1/demo.mp4
```

## Parameter penting (`floor_contact.py`)

| Parameter | Arti | Terlalu banyak pantulan | Terlalu sedikit |
|---|---|---|---|
| `--tol` | jarak dari garis lantai (satuan tinggi bola) | kecilkan | besarkan |
| `--floor-y` | garis lantai manual (piksel) | isi manual | isi manual |
| `--max-gap` | selisih frame maksimum dalam satu kontak | kecilkan | besarkan |
| `--max-dx` | jarak horizontal maksimum dalam satu kontak | kecilkan | besarkan |
