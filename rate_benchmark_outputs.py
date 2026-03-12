#!/usr/bin/env python3
"""Review benchmark outputs and collect human ratings.

This script is intentionally rating-only.
It does NOT edit README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RatingEntry:
    rating: float
    updated_at: str


def rating_key(model_name: str, context_target: int) -> str:
    return f"{model_name}|ctx={context_target}"


def load_ratings(path: Path) -> Dict[str, RatingEntry]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    ratings: Dict[str, RatingEntry] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        rating = value.get("rating")
        if not isinstance(rating, (int, float)):
            continue
        updated_at = str(value.get("updated_at", ""))
        ratings[str(key)] = RatingEntry(rating=float(rating), updated_at=updated_at)

    return ratings


def save_ratings(path: Path, ratings: Dict[str, RatingEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, object]] = {}

    for key, entry in ratings.items():
        # Keep "note" key for compatibility with benchmark script.
        payload[key] = {
            "rating": entry.rating,
            "note": "",
            "updated_at": entry.updated_at,
        }

    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write ratings file at {path}. Choose a user-writable path with --ratings-json."
        ) from exc


def find_benchmark_configs(output_dir: Path) -> List[Tuple[str, int]]:
    configs = set()
    if not output_dir.exists():
        return []

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for ctx_dir in model_dir.iterdir():
            if not ctx_dir.is_dir() or not ctx_dir.name.startswith("ctx_"):
                continue

            try:
                context_target = int(ctx_dir.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue

            run_file = ctx_dir / "outputs" / "run_01.txt"
            if run_file.exists():
                configs.add((model_dir.name, context_target))

    return sorted(configs, key=lambda x: (x[0], x[1]))


def read_output_text(output_dir: Path, model_name: str, context_target: int) -> str:
    run_file = output_dir / model_name / f"ctx_{context_target}" / "outputs" / "run_01.txt"
    if not run_file.exists():
        return "[missing output file]"
    text = run_file.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def prompt_rating(
    model_name: str,
    context_target: int,
    output_text: str,
    rating_min: float,
    rating_max: float,
    existing: Optional[RatingEntry],
) -> Optional[RatingEntry]:
    print("\n" + "=" * 96)
    print(f"Model: {model_name} | Context target: {context_target}")
    print("=" * 96)
    if existing is not None:
        print(f"Current rating: {existing.rating:.2f}")

    print("\nGenerated output (full text):\n")
    print(output_text)

    while True:
        raw = input(
            f"\nEnter rating [{rating_min}-{rating_max}] "
            "(Enter keeps current, s skips): "
        ).strip()

        if raw == "":
            return existing
        if raw.lower() in {"s", "skip"}:
            return existing

        try:
            value = float(raw)
        except ValueError:
            print("Invalid rating. Please enter a number.")
            continue

        if value < rating_min or value > rating_max:
            print("Rating out of range.")
            continue

        return RatingEntry(rating=value, updated_at=datetime.now().isoformat(timespec="seconds"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate benchmark outputs interactively.")
    parser.add_argument("--output-dir", default="llama_server_benchmarks")
    parser.add_argument("--ratings-json", default="ratings.json")
    parser.add_argument("--rating-min", type=float, default=1.0)
    parser.add_argument("--rating-max", type=float, default=5.0)
    parser.add_argument("--model-filter", default="")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.rating_min >= args.rating_max:
        raise ValueError("--rating-min must be smaller than --rating-max")

    if not sys.stdin.isatty():
        print("This script requires an interactive terminal for rating input.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    ratings_path = Path(args.ratings_json).resolve()

    configs = find_benchmark_configs(output_dir)
    if not configs:
        print(f"No benchmark outputs found under {output_dir}", file=sys.stderr)
        return 1

    ratings = load_ratings(ratings_path)

    if args.model_filter:
        configs = [cfg for cfg in configs if args.model_filter in cfg[0]]

    if args.skip_existing:
        configs = [cfg for cfg in configs if rating_key(cfg[0], cfg[1]) not in ratings]

    if not configs:
        print("Nothing to rate (all filtered configs already rated).")
        return 0

    print(f"Found {len(configs)} configurations to rate.")

    for idx, (model_name, context_target) in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}]")
        key = rating_key(model_name, context_target)
        existing = ratings.get(key)
        output_text = read_output_text(output_dir, model_name, context_target)

        chosen = prompt_rating(
            model_name=model_name,
            context_target=context_target,
            output_text=output_text,
            rating_min=args.rating_min,
            rating_max=args.rating_max,
            existing=existing,
        )

        if chosen is not None:
            ratings[key] = chosen

    save_ratings(ratings_path, ratings)
    print(f"\nSaved ratings to {ratings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
