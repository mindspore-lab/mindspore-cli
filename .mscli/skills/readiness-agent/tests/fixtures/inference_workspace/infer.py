#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        if args.model_path and Path(args.model_path).exists():
            print("inference smoke ok")
            return 0
        print("inference smoke missing model")
        return 2

    print("inference entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
