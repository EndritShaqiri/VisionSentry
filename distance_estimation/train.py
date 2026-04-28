from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from distance_estimation.models import DroneRangeHead


DEFAULT_FEATURE_COLUMNS = [
    "score",
    "x_norm",
    "y_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "bbox_diag_px",
    "geometric_distance_m",
    "distance_min_m",
    "distance_max_m",
    "depth_median_m",
    "depth_std_m",
    "depth_valid_fraction",
    "fallback_camera",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight distance-estimation range head from feature CSV.")
    parser.add_argument("--features_csv", type=str, required=True, help="CSV containing detector/ranging features and targets.")
    parser.add_argument("--output", type=str, default="distance_estimation/weights/range_head.pt", help="Checkpoint output path.")
    parser.add_argument("--target_col", type=str, default="distance_target_m", help="Target distance column in meters.")
    parser.add_argument("--range_bin_col", type=str, default="range_bin", help="Optional ordinal label column.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--feature_cols", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS), help="Comma-separated feature columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = [col.strip() for col in args.feature_cols.split(",") if col.strip()]
    df = pd.read_csv(args.features_csv)
    missing = [col for col in feature_columns + [args.target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    range_bin_available = args.range_bin_col in df.columns
    label_to_index = {"close": 0, "medium": 1, "distant": 2}

    features = torch.tensor(df[feature_columns].fillna(0.0).to_numpy(dtype="float32"))
    distances = torch.tensor(df[args.target_col].to_numpy(dtype="float32")).unsqueeze(1)
    if range_bin_available:
        ordinal_labels = torch.tensor(
            [label_to_index.get(str(value).strip().lower(), 1) for value in df[args.range_bin_col]],
            dtype=torch.long,
        )
    else:
        ordinal_labels = torch.zeros((len(df),), dtype=torch.long)

    dataset = TensorDataset(features, distances, ordinal_labels)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DroneRangeHead(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        ordinal_bins=3,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse_loss = nn.SmoothL1Loss()
    nll_loss = nn.GaussianNLLLoss()
    ce_loss = nn.CrossEntropyLoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_features, batch_distances, batch_bins in train_loader:
            optimizer.zero_grad(set_to_none=True)
            mean, log_var, ordinal_logits = model(batch_features)
            variance = torch.exp(log_var)
            loss = mse_loss(mean, batch_distances) + nll_loss(mean, batch_distances, variance)
            if range_bin_available:
                loss = loss + (0.2 * ce_loss(ordinal_logits, batch_bins))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_features)
        train_loss /= max(len(train_dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_distances, batch_bins in val_loader:
                mean, log_var, ordinal_logits = model(batch_features)
                variance = torch.exp(log_var)
                loss = mse_loss(mean, batch_distances) + nll_loss(mean, batch_distances, variance)
                if range_bin_available:
                    loss = loss + (0.2 * ce_loss(ordinal_logits, batch_bins))
                val_loss += float(loss.item()) * len(batch_features)
        val_loss /= max(len(val_dataset), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": feature_columns,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "ordinal_bins": 3,
                    "dropout": args.dropout,
                    "best_val_loss": best_val,
                    "target_col": args.target_col,
                    "range_bin_col": args.range_bin_col if range_bin_available else None,
                },
                output_path,
            )

    print(f"[OK] Saved best checkpoint to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
