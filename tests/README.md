# Test Fixtures

Gunakan folder ini untuk menguji model secara konsisten setelah retraining.

## Folder

- `tests/banana_leaf/`
  Foto daun pisang asli. Isi dengan variasi sehat, sakit ringan, sakit parah, kuning/cokelat, blur ringan, angle dekat, dan background lapangan. Expected: `is_banana_leaf=True`.

- `tests/not_banana_leaf/`
  Foto non-pisang untuk validasi gate. Isi dengan pepaya, kelapa, daun lain, tangan, laptop, screenshot, lantai, dan objek acak. Expected: `is_banana_leaf=False`.

- `hard_negatives/`
  Contoh gagal nyata untuk ikut masuk training negatif. Folder ini tidak dikomit kecuali `.gitkeep`.

- `tests/review/`
  Kasus sulit yang sengaja dipisahkan dari default pass/fail harian. Contoh: daun pisang sakit parah yang masih mental ke `Not Banana Leaf`. Pakai folder ini untuk daftar bahan retraining berikutnya.

## Command

```bash
python tests/test_inference_cases.py
```

Wajib untuk validasi lengkap setelah artifact baru masuk:

```bash
python tests/test_inference_cases.py --require-banana-fixtures
```

Audit kasus sulit:

```bash
python tests/test_inference_cases.py --include-review
```

Minimal fixture yang disarankan:

- 10 foto daun pisang sehat/normal
- 10 foto daun pisang sakit/rusak parah
- 10 foto pepaya
- 10 foto kelapa
- 10 foto tangan/screenshot/objek acak
