"""
Обучение FaceDenoiser — сети для очистки активаций от шума MediaPipe.

Denoising mode (рекомендуемый):
  python train.py --raw data/raw --corrected data/corrected

Autoencoder mode (если нет пар):
  python train.py --raw data/corrected

Дополнительные параметры:
  --epochs 500 --latent 10 --lr 0.001 --batch_size 64

Результат:
  output/face_denoiser.onnx    — модель для Unity Sentis
  output/face_denoiser.pt      — чекпоинт PyTorch
  output/columns.json          — порядок колонок (нужен для инференса)
  output/training_log.json     — лог лоссов
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path

from model import FaceDenoiser
from dataset import create_dataloader


def train(args):
    # --- Данные ---
    loader, dataset = create_dataloader(
        args.raw, args.corrected, batch_size=args.batch_size
    )
    input_dim = dataset.input_dim

    mode = "denoising" if args.corrected else "autoencoder"
    print(f"\nMode: {mode}")
    print(f"Input dim: {input_dim}")
    print(f"Latent dim: {args.latent}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print()

    # --- Модель ---
    model = FaceDenoiser(input_dim=input_dim, latent_dim=args.latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # Нейтраль = все нули в оригинальном пространстве → нормализуем через те же stats
    neutral_raw = torch.zeros(1, input_dim)
    neutral = dataset.norm_stats.normalize(neutral_raw)

    # --- Обучение ---
    log = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_main = 0.0
        total_neutral = 0.0
        n = 0

        for raw_batch, target_batch in loader:
            output = model(raw_batch)

            # Основной лосс: выход должен совпадать с corrected (или с input в AE mode)
            loss_main = criterion(output, target_batch)

            # Лосс нейтрали
            loss_neutral = criterion(model(neutral), neutral)

            loss = loss_main + args.neutral_weight * loss_neutral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_main += loss_main.item()
            total_neutral += loss_neutral.item()
            n += 1

        avg_loss = total_loss / n
        avg_main = total_main / n
        avg_neutral = total_neutral / n

        scheduler.step(avg_loss)

        log.append({
            "epoch": epoch,
            "loss": avg_loss,
            "main": avg_main,
            "neutral": avg_neutral,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"loss: {avg_loss:.6f}  main: {avg_main:.6f}  "
                f"neutral: {avg_neutral:.8f}  lr: {lr:.2e}"
            )

    # --- Сохранение ---
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # PyTorch checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "latent_dim": args.latent,
        "columns": dataset.columns,
    }, out / "face_denoiser.pt")
    print(f"\nSaved: {out / 'face_denoiser.pt'}")

    # ONNX для Unity Sentis
    dummy = torch.randn(1, input_dim)
    onnx_path = str(out / "face_denoiser.onnx")
    torch.onnx.export(
        model, dummy,
        onnx_path,
        input_names=["raw_input"],
        output_names=["cleaned_output"],
        opset_version=15,
    )

    # Embed external weights into single .onnx file (required for Unity Sentis)
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    for tensor in onnx_model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.ClearField("data_location")
            del tensor.external_data[:]
    onnx.save(onnx_model, onnx_path)

    # Cleanup external data file if any
    data_path = out / "face_denoiser.onnx.data"
    if data_path.exists():
        data_path.unlink()

    print(f"Saved: {onnx_path}")

    # Порядок колонок
    with open(out / "columns.json", "w") as f:
        json.dump(dataset.columns, f, indent=2)
    print(f"Saved: {out / 'columns.json'}")

    # Norm stats (min/max per column — нужен для Unity inference)
    dataset.norm_stats.save(str(out / "norm_stats.json"))
    print(f"Saved: {out / 'norm_stats.json'}")

    # Лог
    with open(out / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {out / 'training_log.json'}")

    print(f"\nBest loss: {best_loss:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train FaceRig Denoiser")
    p.add_argument("--raw", type=str, required=True,
                    help="Folder with raw CSV files")
    p.add_argument("--corrected", type=str, default=None,
                    help="Folder with corrected CSV files (omit for autoencoder mode)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--latent", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--neutral_weight", type=float, default=100.0)
    p.add_argument("--output", type=str, default="output")
    train(p.parse_args())
