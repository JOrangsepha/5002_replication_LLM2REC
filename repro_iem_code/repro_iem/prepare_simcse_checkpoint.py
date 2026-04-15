from __future__ import annotations

import argparse
from pathlib import Path

from repro_iem.utils import copy_tokenizer_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy tokenizer artifacts from the base model into an MNTP checkpoint before SimCSE."
    )
    parser.add_argument("--source_model_dir", required=True, help="Base model directory.")
    parser.add_argument("--destination_dir", required=True, help="MNTP checkpoint directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    copied = copy_tokenizer_assets(args.source_model_dir, args.destination_dir)
    destination = Path(args.destination_dir)
    if copied:
        print(f"Copied tokenizer assets into {destination}:")
        for name in copied:
            print(f"  - {name}")
    else:
        print(f"No tokenizer assets matching '*token*' were found under {args.source_model_dir}.")


if __name__ == "__main__":
    main()

