"""Extract inference-only weights from the published IRENE checkpoint.

The checkpoint on the Hub is a PyTorch Lightning training artefact: 770 MB of
which most is optimizer state, learning-rate schedulers, callbacks and loop
state. None of that is needed to run the model, and keeping it means every
deployment downloads and verifies it.

This produces a bundle holding tensors only, which the operational loader can
then read with ``weights_only=True``. Reading the published checkpoint still
requires ``weights_only=False`` because its hyperparameters pickle optimizer
classes, so that happens once, here, rather than on every container start.

It lives in the package rather than under ``tools/`` because the container
runs it on start: an image carries the manifest and no weights, and fills the
bundle itself before the first poll.

Two uses. To fill the bundle this repository already describes, which is what
a fresh clone needs:

    uv run irene-download-artifacts --bundle artifacts/v1

That downloads the checkpoint the manifest names, refuses it unless its SHA-256
is the one the manifest records, extracts the weights, and keeps them only if
they match the manifest too. Pass ``--checkpoint`` to use a copy you already
have instead of downloading.

To create a new bundle, with a new manifest, from a checkpoint of your own:

    uv run irene-download-artifacts \
        --checkpoint model.ckpt --bundle artifacts/v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen

import torch

PREFIX = "model."
WEIGHTS = "irene.pt"
SOURCE_URL = "https://huggingface.co/it4lia/irene"
BLOCK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str, timeout: float) -> None:
    """Fetch one file, keeping it only if it is the file the manifest pinned."""
    digest = hashlib.sha256()
    with urlopen(url, timeout=timeout) as response, target.open("wb") as stream:
        for block in iter(lambda: response.read(BLOCK), b""):
            stream.write(block)
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {url}: the published file is not the pinned one")


def extract(checkpoint_path: Path, target: Path) -> tuple[int, dict]:
    """Write the model tensors of a Lightning checkpoint to target."""
    # weights_only=False is unavoidable here: the published checkpoint pickles
    # optimizer and scheduler classes in its hyperparameters. It is confined to
    # this one offline step, against a checkpoint whose digest is recorded.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = {}
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith(PREFIX):
            continue
        if not torch.is_tensor(value):
            raise ValueError(f"Not a tensor under {key}")
        weights[key[len(PREFIX) :]] = value
    if not weights:
        raise ValueError("No model.* tensors found in the checkpoint state dictionary")
    torch.save(weights, target)
    return len(weights), checkpoint.get("hyper_parameters", {})


def report(count: int, checkpoint: Path, target: Path) -> None:
    before = checkpoint.stat().st_size
    after = target.stat().st_size
    print(f"Extracted {count} tensors")
    print(f"  {before:>14,} bytes  {checkpoint.name}")
    print(f"  {after:>14,} bytes  {target.name}  ({100 * after / before:.1f}%)")


def restore(args: argparse.Namespace, manifest: dict) -> int:
    """Fill an existing bundle, accepting nothing its manifest does not pin."""
    source = manifest["source"]
    declared = manifest["files"]["weights"]
    target = args.bundle / declared["name"]
    if target.is_file() and sha256(target) == declared["sha256"]:
        print(f"{target.name}: already verified")
        return 0

    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=args.bundle) as scratch:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(scratch) / source["name"]
            url = f"{source['url'].rstrip('/')}/resolve/main/{source['name']}"
            print(f"Downloading {url}")
            download(url, checkpoint, source["sha256"], args.timeout)
        elif sha256(checkpoint) != source["sha256"]:
            # The pickle is only ever loaded from the exact file that was pinned.
            raise ValueError(
                f"{checkpoint} is not the checkpoint this bundle was made from; "
                "use a new --bundle directory to make a bundle from it"
            )
        staged = Path(scratch) / declared["name"]
        count, _ = extract(checkpoint, staged)
        actual = sha256(staged)
        if actual != declared["sha256"]:
            raise ValueError(
                f"Extracted weights hash to {actual}, but the manifest records "
                f"{declared['sha256']}. The checkpoint is the pinned one, so the difference "
                f"is how torch {torch.__version__} serialized it. Run this from the locked "
                "environment (uv sync --locked), which is what the manifest was written with."
            )
        report(count, checkpoint, staged)
        os.replace(staged, target)
    print(f"Bundle ready: {args.bundle}")
    return 0


def create(args: argparse.Namespace) -> int:
    """Make a new bundle, manifest included, from a checkpoint supplied by hand."""
    if args.checkpoint is None:
        raise ValueError("A new bundle needs --checkpoint; there is no manifest to say which")
    args.bundle.mkdir(parents=True, exist_ok=True)
    target = args.bundle / WEIGHTS
    if target.exists():
        raise FileExistsError("Use a new bundle directory; existing artifacts are preserved")

    source_digest = sha256(args.checkpoint)
    count, hyper = extract(args.checkpoint, target)
    manifest = {
        "schema_version": 1,
        "version": args.bundle.name,
        "model": "IRENE",
        "architecture": {
            "class": "EncoderDecoder",
            "input_channels": hyper.get("input_channels"),
            "num_blocks": hyper.get("num_blocks"),
            "forecast_steps": hyper.get("forecast_steps"),
            "ensemble_size": hyper.get("ensemble_size"),
            "noisy_decoder": hyper.get("noisy_decoder"),
        },
        "source": {
            "url": args.source_url,
            "name": args.checkpoint.name,
            "sha256": source_digest,
            "license": "BSD-2-Clause",
        },
        # The dBZ transform saturates, so values below this are not representable
        # and the model cannot emit anything smaller. Recorded so the deployment
        # does not have to rediscover it.
        "output_floor_mm_h": 0.036463,
        "files": {"weights": {"name": target.name, "sha256": sha256(target)}},
    }
    (args.bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report(count, args.checkpoint, target)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description="Extract inference-only weights from the published IRENE checkpoint"
    )
    cli.add_argument(
        "--checkpoint",
        type=Path,
        help="A local copy of the checkpoint; omit to download the one the manifest names",
    )
    cli.add_argument("--bundle", type=Path, required=True)
    cli.add_argument("--source-url", default=SOURCE_URL, help="Recorded in a new manifest")
    cli.add_argument("--timeout", type=float, default=60.0)
    args = cli.parse_args(argv)
    manifest_path = args.bundle / "manifest.json"
    try:
        if manifest_path.is_file():
            return restore(args, json.loads(manifest_path.read_text(encoding="utf-8")))
        return create(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Could not prepare the bundle: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
