#!/usr/bin/env python3
"""
Быстрый тестовый reference-HDF5 для v7-переноса.

Берёт несколько FLAME expr-betas как «эмоции», пишет нейтраль + δ каждой
эмоции + НУЛЕВЫЕ активации мышц (заглушка). Готовый файл подаётся в transfer_gui
как Reference HDF5.

  python make_test_reference.py [--out data/reference_test.h5] [--n-muscles 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import make_reference  # noqa: E402


# готовый набор тестовых эмоций (имя → FLAME expr-betas)
TEST_EXPRESSIONS = {
    "smile":      "308:8",
    "frown":      "308:-8",
    "brows_up":   "310:7",
    "brows_down": "310:-7",
    "mouth_open": "311:9",
    "squint":     "312:6,313:-4",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference_test.h5")
    ap.add_argument("--n-muscles", type=int, default=8,
                    help="размер вектора активаций (по нулям)")
    args = ap.parse_args()

    n_m = int(args.n_muscles)
    muscle_names = [f"muscle_{i:02d}" for i in range(n_m)]
    zero_act = [0.0] * n_m
    activations = {name: list(zero_act) for name in TEST_EXPRESSIONS}
    expr_specs = [f"{name}={betas}" for name, betas in TEST_EXPRESSIONS.items()]

    make_reference.build_reference(
        args.out, shape_str="", expr_specs=expr_specs,
        activations=activations, muscle_names=muscle_names)
    print(f"тестовый reference готов: {args.out}")
    print(f"  эмоции: {list(TEST_EXPRESSIONS)}")
    print(f"  активации: {n_m} мышц, все по нулям (заглушка)")


if __name__ == "__main__":
    main()
