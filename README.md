# Nowcast Platform

[![Tests](https://github.com/SInApSE-INPE/nowcast-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/SInApSE-INPE/nowcast-platform/actions/workflows/tests.yml)
[![Documentation](https://github.com/SInApSE-INPE/nowcast-platform/actions/workflows/docs.yml/badge.svg)](https://sinapse-inpe.github.io/nowcast-platform/)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

Operational infrastructure for running gridded precipitation nowcasting models
on a schedule, with auditable provenance.

A nowcasting model is usually published as a checkpoint and a notebook. Running
it in operations takes a great deal more: acquiring observations as they
arrive, refusing incomplete or stale inputs, publishing results that are never
partial, and being able to say afterwards exactly what produced each file. This
repository provides that part once, as a shared runtime, so that a model
contributes only what is specific to it.

**[Documentation](https://sinapse-inpe.github.io/nowcast-platform/)** ·
[Quickstart](https://sinapse-inpe.github.io/nowcast-platform/getting-started/quickstart.html) ·
[Tutorials](https://sinapse-inpe.github.io/nowcast-platform/tutorials/01-your-first-model.html) ·
[Design decisions](docs/adr/README.md)

## Contents

- [Features](#features)
- [Models](#models)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Adding a model](#adding-a-model)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Scope and limitations](#scope-and-limitations)
- [License](#license)

## Features

- **Scheduled acquisition.** A poll is idempotent and locked: it runs when a new
  observation exists, skips a run that is already complete, and waits when the
  input window is incomplete rather than forecasting from a shorter history.
- **Atomic publication.** A run is staged and published by a single rename, so
  a consumer never sees a partial forecast and a complete run is never
  overwritten.
- **Auditable provenance.** Every run carries a `manifest.json` recording its
  inputs, artifacts, processing stages, library versions, device and SHA-256
  checksums.
- **Verified artifacts.** Model weights are pinned by digest and checked before
  each run; a file that does not match is never loaded.
- **A contract that is enforced.** A shared conformance suite of nineteen checks
  runs against each model's real composition, not a mock of it.
- **Isolated models.** Each model has its own lock file and container image, so
  one model's dependencies never constrain another's.
- **A framework-free runtime.** The runtime depends on NumPy and Rasterio. Deep
  learning frameworks belong to the models.

## Models

| Model | Input | Forecast | Output | Operation |
| --- | --- | --- | --- | --- |
| [InceptionV4](models/inceptionv4/) | GSMaP NOW, hourly, 0.1°, 12 steps | 6 hourly leads, deterministic | GeoTIFF, mm/h | Container or systemd timer; JAXA FTP |
| [IRENE](models/irene/) | DPC radar mosaic (Italy), 5 min, 1 km, 6 steps | 12 five-minute leads, ensemble | GeoTIFF, mm/h | Container; DPC open-data API |

Each model directory has its own README covering configuration, artifacts and
deployment.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and, for the containers, Docker
with Compose.

### Run a model operationally

IRENE needs no credentials. Its container downloads and verifies the published
weights on first start, then publishes a forecast for every new radar scan:

```bash
git clone https://github.com/SInApSE-INPE/nowcast-platform.git
cd nowcast-platform/models/irene
mkdir -p outputs && docker compose up -d --build
docker compose logs -f
```

```text
INFO Starting IRENE with a 60-second poll interval
INFO NEW_DATA: processing 2026-11-16T22:50:00+00:00
INFO Published complete run: outputs/irene/dpc_sri/2026/11/16/2250Z
INFO UP_TO_DATE: 2026-11-16T22:50:00+00:00 already completed
```

Forecasts are written under `outputs/<model>/<product>/YYYY/MM/DD/HHMMZ/`, one
GeoTIFF per lead and ensemble member, beside the run manifest. See the
[IRENE README](models/irene/README.md) for operational notes, and the
[InceptionV4 README](models/inceptionv4/README.md) for a deployment that
requires provider credentials.

### Run the test suites

```bash
uv sync --locked
uv run pytest                      # shared runtime

cd models/irene
uv sync --locked
uv run irene-download-artifacts --bundle artifacts/v1
uv run pytest                      # model, including conformance
```

Without the weights, a model's conformance checks are skipped and say so; a
skip means "not run", not "passed".

## How it works

Every run follows the same sequence. The runtime owns the sequence and its
operational guarantees; a deployment supplies the stages.

![The pipeline: source, decoder, optional preprocessing, transform, predictor, inverse transform, optional postprocessing, sink](docs/_static/pipeline-spine.svg)

| Stage | Responsibility |
| --- | --- |
| `Source` | Lists and transfers observations (local directory, FTP, HTTP API). |
| `Decoder` | Recognizes a file, reads it onto a declared grid in declared units. |
| `Preprocess` | Optional input stages, such as a missing-data policy. |
| `Transform` | Maps physical units to model space and back. |
| `Predictor` | The model: an input window in, forecast leads out. |
| `Postprocess` | Optional output stages. |
| `Sink` | Writes each lead and ensemble member to a file. |

The contracts are defined in `nowcast_runtime.ports`. Built-in adapters in
`nowcast_runtime.builtin` are parameterized rather than product-specific, so a
new product is often configuration instead of code. Composition happens in each
model's `entrypoint.py`: the runtime never imports an adapter, and an adapter
never imports the pipeline.

The reasoning behind this structure is recorded in the
[architecture decision records](docs/adr/README.md).

## Adding a model

Generate a project that already passes the conformance suite, then replace its
placeholders one at a time:

```bash
uv run nowcast-new-model --name "MyUNet" --product my_product --units mm/h
uv sync --project models/myunet
uv run --directory models/myunet pytest
```

The generated project contains a persistence predictor, a placeholder bundle,
a Dockerfile and a `README.md` listing each `REPLACE THIS` marker in the order
to address it. What a model declares is a `ModelPlugin`:

```python
PLUGIN = ModelPlugin(
    name="MyUNet",
    slug="myunet",
    version="0.1.0",
    bundle_roles=("checkpoint",),
    requires_runtime=">=0.3,<0.4",
    load=load_model,  # rebuild the network and load the weights
)
```

Three things are always supplied by the model, and the runtime does not infer
them:

1. **The architecture and its loader.** A checkpoint holds weights; code must
   rebuild the network.
2. **The configuration**: domain, source, format and stages.
3. **The artifact bundle**, with a manifest pinning each file by SHA-256.

Conformance is proved by subclassing the shared suite with one hook:

```python
from nowcast_runtime.testing import ConformanceSuite


class TestMyModel(ConformanceSuite):
    product = "my_product"

    def build(self, tmp_path):
        return compose(args_for(tmp_path))
```

The tutorials walk through the whole process, from
[a first model](https://sinapse-inpe.github.io/nowcast-platform/tutorials/01-your-first-model.html)
to
[porting a published one](https://sinapse-inpe.github.io/nowcast-platform/tutorials/06-porting-a-published-model.html).

## Repository layout

```text
├── packages/nowcast-runtime/   # shared runtime: ports, pipeline, scheduler,
│                               # built-in adapters, conformance suite, scaffold
├── models/
│   ├── inceptionv4/            # GSMaP NOW, six hourly leads
│   └── irene/                  # DPC radar, twelve five-minute ensemble leads
├── docs/                       # documentation source and decision records
└── pyproject.toml              # uv workspace (packages only) and lint rules
```

`packages/*` form one uv workspace with a shared lock. Models are deliberately
not workspace members: each keeps its own `uv.lock`, so a CPU-only deployment
and a CUDA one can coexist
([ADR 0002](docs/adr/0002-workspace-and-model-isolation.md)).

## Development

```bash
uv run ruff check .        # one lint configuration for the whole repository
uv run pytest              # runtime tests, from the repository root

# documentation
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Continuous integration runs the runtime tests on Python 3.11, 3.12 and 3.13,
and each model against its own lock with its real artifact bundle. Commit
messages follow [Conventional Commits](https://www.conventionalcommits.org/).

## Scope and limitations

The scope is intentionally narrow
([ADR 0001](docs/adr/0001-pipeline-shape-scope.md)):

> Gridded geospatial sequence-to-sequence forecasting, with scheduled batch
> publication and auditable provenance.

The runtime verifies contracts: units, grid alignment, window spacing, output
shape and artifact checksums. It does not, and cannot, verify scientific
validity. Whether a model has skill on a given product or region remains the
responsibility of that model's own validation.

## License

A license has not yet been selected for this repository. Third-party code
vendored by a model keeps its original license, included alongside it (for
example, `models/irene/src/irene_nowcast/_vendor/LICENSE`).

## Acknowledgements

Developed at SInApSE, National Institute for Space Research (INPE), Brazil. InceptionV4 forecasts use GSMaP NOW data from JAXA. IRENE is a
model by Fondazione Bruno Kessler, run here on radar data from the Italian
Dipartimento della Protezione Civile.
