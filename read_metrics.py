"""Read and summarise per-client metrics saved as CSV."""

import argparse
from pathlib import Path

import numpy as np


def load_metrics(path: Path) -> None:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    print(f"Loaded {path}")
    print(f"Shape: {arr.shape} (expected num_clients x 5)")
    client_id = arr[:, 0].astype(int)
    loss = arr[:, 1]
    acc = arr[:, 2]
    data_size = arr[:, 3]
    grad_norm = arr[:, 4]

    print("First 10 rows (client_id, loss, acc, data_size, grad_norm):")
    print(arr[:10])
    print("Nanmean (ignoring nan):")
    print("  loss:", np.nanmean(loss))
    print("  acc :", np.nanmean(acc))
    print("  data:", np.nanmean(data_size))
    print("  grad:", np.nanmean(grad_norm))

    # Optional: list active clients (non-nan loss)
    active = client_id[~np.isnan(loss)]
    print(f"Active clients this round: {len(active)} / {len(client_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and summarise per-client metrics saved as CSV"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to metrics file, e.g. dataset/round_11_metrics.csv",
    )
    args = parser.parse_args()
    load_metrics(args.path)


if __name__ == "__main__":
    main()
