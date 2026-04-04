#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        config_ok = bool(args.config and Path(args.config).exists())
        dataset_ok = bool(args.dataset and Path(args.dataset).exists())
        model_ok = bool(args.model_path and Path(args.model_path).exists())
        if config_ok and dataset_ok and model_ok:
            print("training smoke ok")
            return 0
        print("training smoke missing asset")
        return 2

    print("training entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
