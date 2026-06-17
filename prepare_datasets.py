"""Siapkan folder `datasets/` two-stage: 8 kelas pisang + 1 kelas negatif "Not Banana Leaf".

Latar belakang:
    Saat sidang, sistem salah membaca daun pepaya, daun kelapa, daun lain, lantai,
    tangan, dan kaki sebagai PENYAKIT pisang. Heuristik OOD berbasis ImageNet tidak
    cukup andal. Solusi: latih kelas negatif eksplisit agar model belajar batas
    "ini daun pisang / bukan".

Yang dilakukan script ini:
    1. Download dataset pisang dari Kaggle -> susun 8 folder kelas "Augmented Banana ...".
    2. Download beberapa dataset negatif (daun lain + bukan-daun) -> sampling SEIMBANG
       ke satu folder "Not Banana Leaf" (default 4000 gambar).
    3. Tambahkan semua gambar di hard_negatives/ (contoh gagal nyata: tangan, laptop,
       screenshot, daun non-pisang) ke kelas negatif.
    4. Cetak ringkasan jumlah gambar per kelas.

Prasyarat:
    - kaggle CLI terpasang & kredensial di ~/.kaggle/kaggle.json (sudah ada di mesin ini).
    - Jalankan dari root repo:  python prepare_datasets.py
    - Opsi:  --neg-only (lewati pisang),  --no-download (pakai staging yang sudah ada),
             --neg-total N (paksa jumlah gambar negatif dari Kaggle; hard_negatives
             selalu ditambahkan di luar kuota).

Catatan transparansi (BUKAN seluruh isi dataset dipakai):
    - PlantVillage (4.4GB, 3 salinan redundan) sengaja TIDAK dipakai; diganti PlantDoc
      (foto lapangan, lebih mirip kondisi nyata).
    - Setiap sumber negatif di-SAMPLING acak (seed tetap) sampai kuota, sisanya dibuang.
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
STAGING = ROOT / ".dataset_staging"           # area unduh sementara (boleh dihapus)
HARD_NEGATIVES = ROOT / "hard_negatives"      # contoh gagal nyata, lokal-only
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 8 kelas pisang (harus sama persis dengan artifacts/labels.json).
BANANA_LABELS = [
    "Augmented Banana Black Sigatoka Disease",
    "Augmented Banana Bract Mosaic Virus Disease",
    "Augmented Banana Cordana Disease",
    "Augmented Banana Healthy Leaf",
    "Augmented Banana Insect Pest Disease",
    "Augmented Banana Moko Disease",
    "Augmented Banana Panama Disease",
    "Augmented Banana Yellow Sigatoka Disease",
]
NEG_LABEL = "Not Banana Leaf"

# Sumber dataset pisang di Kaggle.
#   main    -> berisi 7 kelas "Augmented Banana ..." (tanpa Cordana)
#   cordana -> BananaLSD, sumber kelas Cordana
BANANA_SOURCES = {
    "main": "sujaykapadnis/banana-disease-recognition-dataset",
    "cordana": "shifatearman/bananalsd",
}

# Sumber NEGATIF. `weight` = proporsi kuota negatif yang diambil dari sumber ini.
# Dikelompokkan agar negatif fokus ke bahan uji utama:
#   daun pepaya + daun kelapa dominan, sumber lain hanya pendukung agar gate tetap umum.
NEG_SOURCES = [
    # --- Daun non-pisang utama untuk pengujian (~70%) ---
    {"slug": "ajithdari/papaya-leaf-disease-dataset",                    "weight": 0.35, "group": "daun-pepaya"},
    {"slug": "shravanatirtha/coconut-leaf-dataset-for-pest-identification", "weight": 0.35, "group": "daun-kelapa"},
    # --- Daun non-pisang tambahan agar gate tidak hafal hanya 2 spesies (~10%) ---
    {"slug": "nirmalsankalana/plantdoc-dataset",                        "weight": 0.10, "group": "daun-multispesies"},
    # --- Bukan daun pendukung: tangan/objek/scene (~20%) ---
    {"slug": "shyambhu/hands-and-palm-images-dataset",                  "weight": 0.08, "group": "tangan"},
    {"slug": "prasunroy/natural-images",                                "weight": 0.07, "group": "objek-acak"},
    {"slug": "itsahmad/indoor-scenes-cvpr-2019",                        "weight": 0.05, "group": "scene-lantai",
     "enabled": True},
]

DEFAULT_NEG_TOTAL = 4000


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def run_kaggle_download(slug: str, dest: Path) -> bool:
    """Unduh & ekstrak satu dataset Kaggle ke `dest`. Idempoten: lewati jika sudah ada isi."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"  [skip download] {slug} -> {dest} (sudah ada isi)")
        return True
    print(f"  [download] {slug} -> {dest}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  [GAGAL] {slug}: {exc}")
        return False


def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]


def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def find_class_dir(staging: Path, label: str) -> Path | None:
    """Cari folder di dalam staging yang paling cocok dengan nama `label`.

    Cocokkan via beberapa kata kunci penyakit (sigatoka, moko, panama, dst) agar tahan
    terhadap perbedaan penamaan folder antar versi dataset.
    """
    # kata kunci unik per kelas
    key = label.replace("Augmented Banana", "").replace("Disease", "").replace("Leaf", "").strip()
    key_n = norm(key)
    candidates: list[tuple[int, Path]] = []
    for d in staging.rglob("*"):
        if not d.is_dir():
            continue
        dn = norm(d.name)
        if not dn:
            continue
        # skor: cocok penuh > mengandung kata kunci
        if dn == norm(label):
            candidates.append((3, d))
        elif key_n and key_n in dn:
            candidates.append((2, d))
        elif key_n and dn in key_n and len(dn) >= 4:
            candidates.append((1, d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], len(list_images(t[1]))), reverse=True)
    return candidates[0][1]


def copy_sample(images: list[Path], n: int, dest: Path, prefix: str, rng: random.Random) -> int:
    """Salin maksimal `n` gambar (acak) ke `dest` dengan nama berprefiks (anti-bentrok)."""
    dest.mkdir(parents=True, exist_ok=True)
    chosen = images if len(images) <= n else rng.sample(images, n)
    for i, src in enumerate(chosen):
        out = dest / f"{prefix}_{i:05d}{src.suffix.lower()}"
        try:
            shutil.copy2(src, out)
        except OSError:
            pass
    return len(chosen)


# ---------------------------------------------------------------------------
# Tahap 1: pisang
# ---------------------------------------------------------------------------
def build_banana(no_download: bool) -> dict[str, int]:
    print("\n=== Tahap 1: Dataset pisang (8 kelas) ===")
    counts: dict[str, int] = {}
    main_stage = STAGING / "banana_main"
    cordana_stage = STAGING / "banana_cordana"

    if not no_download:
        run_kaggle_download(BANANA_SOURCES["main"], main_stage)
        run_kaggle_download(BANANA_SOURCES["cordana"], cordana_stage)

    for label in BANANA_LABELS:
        # Cordana diambil dari BananaLSD; sisanya dari dataset utama.
        search_root = cordana_stage if "Cordana" in label else main_stage
        src_dir = find_class_dir(search_root, label)
        dest = DATASETS / label
        dest.mkdir(parents=True, exist_ok=True)
        if src_dir is None:
            print(f"  [!] folder sumber untuk '{label}' TIDAK ditemukan di {search_root}")
            counts[label] = len(list_images(dest))
            continue
        imgs = list_images(src_dir)
        rng = random.Random(SEED)
        n = copy_sample(imgs, len(imgs), dest, prefix=norm(label)[:12], rng=rng)
        print(f"  [{label}] <- {src_dir.relative_to(STAGING)} ({n} gambar)")
        counts[label] = len(list_images(dest))
    return counts


# ---------------------------------------------------------------------------
# Tahap 2: negatif
# ---------------------------------------------------------------------------
def build_negative(neg_total: int, no_download: bool) -> int:
    print(f"\n=== Tahap 2: Kelas negatif '{NEG_LABEL}' (target ~{neg_total} gambar) ===")
    dest = DATASETS / NEG_LABEL
    if dest.exists():
        shutil.rmtree(dest)  # selalu bangun ulang agar proporsi konsisten
    dest.mkdir(parents=True, exist_ok=True)

    active = [s for s in NEG_SOURCES if s.get("enabled", True)]
    total_w = sum(s["weight"] for s in active)
    grand_total = 0
    for s in active:
        stage = STAGING / ("neg_" + s["group"])
        if not no_download:
            ok = run_kaggle_download(s["slug"], stage)
            if not ok:
                print(f"  [lewati] {s['group']} (download gagal)")
                continue
        imgs = list_images(stage)
        if not imgs:
            print(f"  [lewati] {s['group']} (tidak ada gambar di {stage})")
            continue
        quota = max(1, round(neg_total * s["weight"] / total_w))
        rng = random.Random(SEED + hash(s["group"]) % 1000)
        n = copy_sample(imgs, quota, dest, prefix=s["group"], rng=rng)
        print(f"  [{s['group']:18s}] tersedia {len(imgs):5d} -> diambil {n} (kuota {quota})")
        grand_total += n

    hard_imgs = list_images(HARD_NEGATIVES) if HARD_NEGATIVES.exists() else []
    if hard_imgs:
        rng = random.Random(SEED + 9999)
        n = copy_sample(hard_imgs, len(hard_imgs), dest, prefix="hardnegative", rng=rng)
        print(f"  [{'hard-negatives':18s}] tersedia {len(hard_imgs):5d} -> ditambahkan {n} (di luar kuota)")
        grand_total += n
    else:
        print(f"  [hard-negatives   ] tidak ada gambar di {HARD_NEGATIVES} (opsional)")
    print(f"  TOTAL negatif: {grand_total} gambar")
    return grand_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Siapkan datasets/ untuk pipeline two-stage")
    ap.add_argument("--neg-only", action="store_true", help="lewati tahap pisang")
    ap.add_argument("--no-download", action="store_true", help="pakai staging yang sudah ada")
    ap.add_argument("--neg-total", type=int, default=DEFAULT_NEG_TOTAL,
                    help="jumlah gambar negatif dari Kaggle (default: 4000)")
    ap.add_argument("--keep-staging", action="store_true", help="jangan hapus .dataset_staging di akhir")
    args = ap.parse_args()

    DATASETS.mkdir(exist_ok=True)
    banana_counts: dict[str, int] = {}

    if not args.neg_only:
        banana_counts = build_banana(args.no_download)
    else:
        for label in BANANA_LABELS:
            banana_counts[label] = len(list_images(DATASETS / label))

    # Tentukan target negatif dari Kaggle. Hard negatives lokal selalu ditambahkan
    # di luar kuota agar contoh gagal nyata tidak terbuang.
    valid = [c for c in banana_counts.values() if c > 0]
    avg = round(sum(valid) / len(valid)) if valid else 1000
    neg_total = args.neg_total or DEFAULT_NEG_TOTAL or avg
    build_negative(neg_total, args.no_download)

    # Ringkasan
    print("\n=== Ringkasan datasets/ ===")
    grand = 0
    for label in BANANA_LABELS + [NEG_LABEL]:
        n = len(list_images(DATASETS / label))
        grand += n
        print(f"  {label:45s} : {n}")
    print(f"  {'TOTAL':45s} : {grand}")
    print(f"\nSelesai. Folder siap di: {DATASETS}")
    print("Langkah berikutnya: jalankan train-collabs.ipynb untuk training two-stage.")

    if not args.keep_staging and STAGING.exists():
        print(f"(Hapus staging manual bila perlu: rm -rf {STAGING})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
