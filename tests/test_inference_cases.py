"""Batch test untuk Banana Doctor two-stage inference.

Taruh fixture di:
  - tests/banana_leaf/      -> expected is_banana_leaf=True
  - tests/not_banana_leaf/  -> expected is_banana_leaf=False
  - tests/review/           -> kasus sulit, hanya dipakai jika --include-review
  - hard_negatives/         -> expected is_banana_leaf=False

Script ini sengaja tidak memakai pytest agar mudah dijalankan langsung di Colab
atau terminal lokal:
  python tests/test_inference_cases.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_artifacts, predict_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Case:
    path: Path
    expected_banana: bool
    group: str


def iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMG_EXT:
        return [path]
    if not path.exists():
        return []
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def collect_cases(include_hard_negatives: bool = True) -> list[Case]:
    cases: list[Case] = []

    builtins = [
        (Path("tests/daun_pepaya.jpeg"), False, "builtin-negative"),
        (Path("tests/daun_kelapa.jpeg"), False, "builtin-negative"),
        (Path("tests/tangan.jpeg"), False, "builtin-negative"),
    ]
    for path, expected, group in builtins:
        if path.exists():
            cases.append(Case(path, expected, group))

    for path in iter_images(Path("tests/not_banana_leaf")):
        cases.append(Case(path, False, "not_banana_leaf"))

    if include_hard_negatives:
        for path in iter_images(Path("hard_negatives")):
            cases.append(Case(path, False, "hard_negatives"))

    for path in iter_images(Path("tests/banana_leaf")):
        cases.append(Case(path, True, "banana_leaf"))

    return cases


def collect_review_cases() -> list[Case]:
    cases: list[Case] = []
    for path in iter_images(Path("tests/review/banana_leaf_false_negative")):
        cases.append(Case(path, True, "review_banana_fn"))
    for path in iter_images(Path("tests/review/not_banana_false_positive")):
        cases.append(Case(path, False, "review_not_banana_fp"))
    return cases


def fmt_prob(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch inference test cases")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah case")
    parser.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="jangan scan folder hard_negatives/",
    )
    parser.add_argument(
        "--require-banana-fixtures",
        action="store_true",
        help="gagal jika tests/banana_leaf/ kosong",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="ikutkan kasus sulit di tests/review/ sebagai expected pass/fail reguler",
    )
    args = parser.parse_args()

    artifacts, err = load_artifacts()
    if err:
        print(f"ERROR load_artifacts: {err}")
        return 2
    if artifacts is None:
        print("ERROR artifacts belum tersedia")
        return 2

    cases = collect_cases(include_hard_negatives=not args.no_hard_negatives)
    if args.include_review:
        cases.extend(collect_review_cases())
    if args.limit is not None:
        cases = cases[:args.limit]

    banana_fixture_count = sum(1 for c in cases if c.group == "banana_leaf")
    if args.require_banana_fixtures and banana_fixture_count == 0:
        print("ERROR tests/banana_leaf/ kosong. Tambahkan fixture daun pisang nyata.")
        return 2

    if not cases:
        print("Tidak ada test case gambar.")
        return 2

    print("Model mode :", artifacts.get("mode"))
    print("Labels     :", len(artifacts.get("all_labels", [])))
    print("Models     :", artifacts.get("model_names"))
    if artifacts.get("banana_gate_config"):
        print("Threshold  :", artifacts["banana_gate_config"].get("banana_threshold"))
    print()

    failures: list[tuple[Case, dict[str, object]]] = []
    group_totals: dict[str, int] = {}
    group_passed: dict[str, int] = {}

    header = (
        f"{'OK':2s}  {'expected':8s} {'actual':8s} "
        f"{'banana_p':>8s} {'conf':>8s}  {'group':18s}  label / file"
    )
    print(header)
    print("-" * len(header))

    for case in cases:
        group_totals[case.group] = group_totals.get(case.group, 0) + 1
        try:
            image = Image.open(case.path)
            result = predict_image(artifacts, image)
        except Exception as exc:
            result = {
                "label": f"ERROR: {exc}",
                "is_banana_leaf": None,
                "confidence": None,
                "banana_probability": None,
            }

        actual = result.get("is_banana_leaf")
        passed = actual is case.expected_banana
        if passed:
            group_passed[case.group] = group_passed.get(case.group, 0) + 1
        else:
            failures.append((case, result))

        expected_text = "banana" if case.expected_banana else "not"
        actual_text = (
            "banana" if actual is True
            else "not" if actual is False
            else "error"
        )
        marker = "OK" if passed else "NO"
        print(
            f"{marker:2s}  {expected_text:8s} {actual_text:8s} "
            f"{fmt_prob(result.get('banana_probability')):>8s} "
            f"{fmt_prob(result.get('confidence')):>8s}  "
            f"{case.group:18s}  {result.get('label')} / {case.path}"
        )

    print("\nRingkasan:")
    total = len(cases)
    passed_total = total - len(failures)
    print(f"  PASS {passed_total}/{total}")
    for group in sorted(group_totals):
        passed = group_passed.get(group, 0)
        print(f"  {group:18s}: {passed}/{group_totals[group]}")

    if banana_fixture_count == 0:
        print("\nCatatan: tests/banana_leaf/ masih kosong; tambahkan foto daun pisang nyata untuk uji false negative.")

    if failures:
        print("\nFailures:")
        for case, result in failures:
            print(
                f"  - {case.path}: expected is_banana_leaf={case.expected_banana}, "
                f"got {result.get('is_banana_leaf')} ({result.get('label')})"
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
