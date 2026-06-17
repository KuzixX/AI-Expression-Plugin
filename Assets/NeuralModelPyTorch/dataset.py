"""
Загрузка CSV для обучения denoising-сети.

Нормализация: per-column min-max → [-1..1] для ВСЕХ колонок.
min/max сохраняются в norm_stats.json для денормализации в Unity.

Использование:
  loader, dataset = create_dataloader("data/raw", "data/corrected", batch_size=64)
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


def load_csv(path: str) -> tuple[list[str], torch.Tensor]:
    """Загружает CSV, возвращает (column_names, data_tensor)."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]

    # Убираем timestamp (первая колонка)
    columns = header[1:]
    data = torch.tensor([row[1:] for row in rows], dtype=torch.float32)
    return columns, data


class NormStats:
    """Per-column min-max для нормализации в [-1..1]."""

    def __init__(self, data: torch.Tensor, columns: list[str]):
        self.columns = columns
        # min/max по всем строкам для каждой колонки
        self.col_min = data.min(dim=0).values  # shape: (num_features,)
        self.col_max = data.max(dim=0).values

        # Избегаем деления на 0: если min == max, ставим range = 1
        self.col_range = self.col_max - self.col_min
        self.col_range[self.col_range < 1e-8] = 1.0

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """[min..max] → [-1..1]"""
        return (data - self.col_min) / self.col_range * 2.0 - 1.0

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """[-1..1] → [min..max]"""
        return (data + 1.0) / 2.0 * self.col_range + self.col_min

    def save(self, path: str):
        """Сохраняет min/max в JSON для использования в Unity."""
        stats = {}
        for i, col in enumerate(self.columns):
            stats[col] = {
                "min": float(self.col_min[i]),
                "max": float(self.col_max[i]),
            }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @staticmethod
    def load(path: str) -> "NormStats":
        """Загружает из JSON."""
        with open(path, "r") as f:
            stats = json.load(f)
        columns = list(stats.keys())
        mins = torch.tensor([stats[c]["min"] for c in columns])
        maxs = torch.tensor([stats[c]["max"] for c in columns])
        ns = NormStats.__new__(NormStats)
        ns.columns = columns
        ns.col_min = mins
        ns.col_max = maxs
        ns.col_range = maxs - mins
        ns.col_range[ns.col_range < 1e-8] = 1.0
        return ns


class FaceDenoisingDataset(Dataset):
    def __init__(self, raw_dir: str, corrected_dir: str):
        raw_path = Path(raw_dir)
        corrected_path = Path(corrected_dir)

        raw_files = sorted(raw_path.glob("*.csv"))
        if not raw_files:
            raise FileNotFoundError(f"No CSV files in {raw_dir}")

        all_raw = []
        all_corrected = []
        self.columns = None

        for raw_file in raw_files:
            corrected_file = corrected_path / raw_file.name
            if not corrected_file.exists():
                print(f"  WARNING: no corrected pair for {raw_file.name}, skipping")
                continue

            raw_cols, raw_data = load_csv(str(raw_file))
            cor_cols, cor_data = load_csv(str(corrected_file))

            if self.columns is None:
                self.columns = raw_cols

            assert raw_cols == cor_cols, f"Column mismatch: {raw_file.name}"
            assert raw_data.shape[0] == cor_data.shape[0], \
                f"Frame count mismatch in {raw_file.name}: raw={raw_data.shape[0]}, corrected={cor_data.shape[0]}"

            all_raw.append(raw_data)
            all_corrected.append(cor_data)
            print(f"  Loaded pair: {raw_file.name} ({raw_data.shape[0]} frames)")

        if not all_raw:
            raise FileNotFoundError("No matching raw/corrected CSV pairs found")

        raw_cat = torch.cat(all_raw, dim=0)
        cor_cat = torch.cat(all_corrected, dim=0)

        # Compute norm stats from ALL data (raw + corrected combined)
        combined = torch.cat([raw_cat, cor_cat], dim=0)
        self.norm_stats = NormStats(combined, self.columns)

        self.raw = self.norm_stats.normalize(raw_cat)
        self.corrected = self.norm_stats.normalize(cor_cat)
        self.input_dim = self.raw.shape[1]

        print(f"Total: {len(self.raw)} frame pairs, {self.input_dim} features")

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return self.raw[idx], self.corrected[idx]


class SingleCSVDataset(Dataset):
    """Автоэнкодер на одних данных."""
    def __init__(self, csv_dir: str):
        csv_path = Path(csv_dir)
        csv_files = sorted(csv_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {csv_dir}")

        all_data = []
        self.columns = None

        for f in csv_files:
            cols, data = load_csv(str(f))
            if self.columns is None:
                self.columns = cols
            all_data.append(data)
            print(f"  Loaded: {f.name} ({data.shape[0]} frames)")

        raw = torch.cat(all_data, dim=0)

        # Per-column min-max normalization
        self.norm_stats = NormStats(raw, self.columns)
        self.data = self.norm_stats.normalize(raw)
        self.input_dim = self.data.shape[1]

        print(f"Total: {len(self.data)} frames, {self.input_dim} features")
        print(f"Value range after norm: [{self.data.min():.3f}, {self.data.max():.3f}]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


def create_dataloader(
    raw_dir: str,
    corrected_dir: Optional[str] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tuple:
    if corrected_dir and os.path.isdir(corrected_dir):
        dataset = FaceDenoisingDataset(raw_dir, corrected_dir)
    else:
        print("No corrected dir — falling back to autoencoder mode")
        dataset = SingleCSVDataset(raw_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
