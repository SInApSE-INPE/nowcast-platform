"""Fill the weights bundle; the same thing `irene-download-artifacts` runs.

    uv run python tools/prepare_artifacts.py --bundle artifacts/v1

The work lives in `irene_nowcast.weights`, inside the package, because the
container prepares its own bundle on start and an image carries no `tools/`.
"""

from irene_nowcast.weights import main

if __name__ == "__main__":
    raise SystemExit(main())
