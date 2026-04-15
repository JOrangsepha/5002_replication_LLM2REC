from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def reservoir_sample_csv(
    input_csv: Path,
    sample_size: int,
    seed: int,
) -> Tuple[List[str], List[List[str]], int]:
    """Uniformly sample exactly sample_size rows from CSV using reservoir sampling."""
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    rng = random.Random(seed)
    reservoir: List[List[str]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        total = 0
        for row in reader:
            total += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                # random integer in [0, total-1]
                j = rng.randrange(total)
                if j < sample_size:
                    reservoir[j] = row

    if total == 0:
        raise ValueError(f"No data rows found in {input_csv}")

    if sample_size >= total:
        # Reservoir already holds all rows in this case.
        sample_size = total

    return header, reservoir[:sample_size], total


def write_csv(output_csv: Path, header: List[str], rows: List[List[str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible small CSFT train/valid CSVs from AmazonMix-6."
    )
    parser.add_argument(
        "--train_input",
        type=Path,
        default=Path("data/AmazonMix-6/5-core/train/AmazonMix-6.csv"),
    )
    parser.add_argument(
        "--valid_input",
        type=Path,
        default=Path("data/AmazonMix-6/5-core/valid/AmazonMix-6.csv"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/AmazonMix-6-mini/5-core"),
        help="Output root that will contain train/ and valid/ subfolders.",
    )
    parser.add_argument(
        "--train_rows",
        type=int,
        default=50000,
        help="Number of sampled train rows.",
    )
    parser.add_argument(
        "--valid_rows",
        type=int,
        default=10000,
        help="Number of sampled valid rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_header, train_rows, train_total = reservoir_sample_csv(
        args.train_input, args.train_rows, args.seed
    )
    valid_header, valid_rows, valid_total = reservoir_sample_csv(
        args.valid_input, args.valid_rows, args.seed + 1
    )

    train_out = args.output_root / "train" / "AmazonMix-6-mini.csv"
    valid_out = args.output_root / "valid" / "AmazonMix-6-mini.csv"

    write_csv(train_out, train_header, train_rows)
    write_csv(valid_out, valid_header, valid_rows)

    print("Done.")
    print(f"Train: {args.train_input} ({train_total} rows) -> {train_out} ({len(train_rows)} rows)")
    print(f"Valid: {args.valid_input} ({valid_total} rows) -> {valid_out} ({len(valid_rows)} rows)")


if __name__ == "__main__":
    main()
