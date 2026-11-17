# IRENE — ensemble radar nowcasting on the DPC mosaic

[IRENE](https://huggingface.co/it4lia/irene) is a ConvGRU ensemble nowcaster
from Fondazione Bruno Kessler, trained on the Italian DPC radar mosaic and
published under BSD 2-Clause. This project runs it on the shared
`nowcast-runtime`: six consecutive five-minute rain-rate scans in, twelve
five-minute leads out (one hour), one GeoTIFF per lead and ensemble member, in
mm/h. Scans come from the DPC open-data radar API, and the project ships as a
container that forecasts from every new one.

Acquisition, preprocessing, export and the run manifest come from the runtime.
This directory holds only what is specific to IRENE. How the port was done, and
what it cost the platform, is written up in
[the porting tutorial](../../docs/tutorials/06-porting-a-published-model.md).

## Run it operationally, in Docker

### From the published image

Nothing to clone or build. The image is published to the GitHub Container
Registry on every change to `main`:

```bash
mkdir -p outputs
docker run -d --name irene --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    -v irene-artifacts:/app/artifacts/v1 \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/sinapse-inpe/irene-nowcast:latest
docker logs -f irene
```

The named volume keeps the weights, so they are downloaded once rather than on
every new container. `latest` follows `main`; to pin a deployment, use the
model version (`:0.1.0`) or a commit (`:sha-<short>`).

### From this repository

From `models/irene`:

```bash
mkdir -p outputs && docker compose up -d --build
docker compose logs -f
```

That is the whole deployment. On first start the container downloads the
published checkpoint and extracts the weights into `artifacts/v1` (about a
minute, 770 MB down, 257 MB kept); after that it asks the DPC API once a minute
whether a new scan exists and publishes a forecast for each one:

```text
INFO Starting IRENE with a 60-second poll interval
irene.pt: already verified
INFO NEW_DATA: processing 2026-11-16T22:50:00+00:00
INFO Acquired 6 consecutive inputs ending at 2026-11-16T22:50:00+00:00
INFO Published complete run: outputs/irene/dpc_sri/2026/11/16/2250Z
INFO UP_TO_DATE: 2026-11-16T22:50:00+00:00 already completed
```

| Log line | Meaning |
| --- | --- |
| `NEW_DATA` | A scan newer than the last forecast exists; a run starts. |
| `UP_TO_DATE` | The newest scan already has a complete forecast. |
| `WAITING` | One of the six scans is not available yet, or the newest is over an hour old. Nothing is published; the next poll tries again. |
| `Scheduled poll failed` | The API or the network failed after retries. The next poll tries again. |

A scan appears about seven minutes after its nominal time, and a run on four
CPU threads takes under twenty seconds, so a forecast is on disk roughly eight
minutes after the last observation it used.

Things to know before leaving it running:

- **Create `outputs/` yourself**, as above. A bind-mount directory Docker has
  to create belongs to root, and the service does not run as root.
- **It runs as UID/GID 1000**, so files land on the host owned by the first
  user. If yours differs, set `IRENE_UID` and `IRENE_GID` (in the shell, or in
  a `.env` file beside `compose.yaml`).
- **Nothing deletes old forecasts.** A run is about 0.4 MB with the default
  domain and four members, so a day is roughly 100 MB. Rotate `outputs/` with
  whatever the host already uses.
- **`NOWCAST_POLL_INTERVAL_SECONDS`** changes the polling cadence (default 60).
- **No terms of use are stated for the API.** It needs no account, but the
  documentation found for it gives neither a licence nor a rate limit. One
  small request a minute is what this service sends when nothing is new.

`docker compose down` stops it; an interrupted run leaves nothing behind,
because a run is published by one rename once every file exists.

## Project layout

```text
models/irene/
├── config/default.json          # domain, missing-data policy, members, seed
├── src/irene_nowcast/
│   ├── entrypoint.py            # the only file that knows both model and runtime
│   ├── model.py                 # predictor and the reflectivity transform
│   ├── source.py                # the DPC open-data radar API
│   ├── decoder.py               # the DPC GeoTIFF and NetCDF formats
│   ├── weights.py               # download the published checkpoint, extract the weights
│   ├── stages.py                # Threshold: bounds the output to what dBZ can represent
│   └── _vendor/                 # upstream architecture, verbatim, with its licence
├── artifacts/v1/manifest.json   # digests of the weights; the weights are not in Git
├── Dockerfile, compose.yaml     # the operational service
├── tools/make_sample_inputs.py  # synthetic scans for a smoke test
└── tests/                       # conformance, the API source, the decoders
```

## Get the weights

The container does this by itself. Outside Docker, the weights are not in the
repository: they are extracted from the checkpoint FBK publishes at
<https://huggingface.co/it4lia/irene>. From `models/irene`:

```bash
uv sync --locked
uv run irene-download-artifacts --bundle artifacts/v1
```

That downloads `model.ckpt` (770 MB), refuses it unless its SHA-256 is the one
`artifacts/v1/manifest.json` records, keeps the tensors only, and writes
`artifacts/v1/irene.pt` (257 MB) only if that matches the manifest too. Running
it again is a no-op. If you already have the checkpoint, add
`--checkpoint /path/to/model.ckpt` to skip the download.

Two thirds of the published file is optimizer and training-loop state, and it
can only be read with `weights_only=False`. Doing the extraction once, against
a pinned digest, is what lets every later run load tensors with
`weights_only=True`.

Run it from the locked environment. The digest in the manifest is the one
torch from `uv.lock` produces; another torch build can serialize the same
tensors to different bytes, and the tool then stops with both digests rather
than keep a file the manifest does not describe.

## Checks

```bash
uv run pytest
uv run ruff check src tests tools
```

Nineteen conformance checks run, beside tests of the API source (against a
scripted transport, so no network) and of the decoders. Without
`artifacts/v1/irene.pt` the conformance checks **skip**, naming the missing
file; read that as "not run", never as "passed".

## Run it by hand

One forecast from the newest scans the API has:

```bash
uv run irene-nowcast --config config/default.json --source dpc --output-dir outputs
```

Add `--run-time 2026-11-16T22:35:00Z` for a past window the API still serves,
or `--scheduled` for exactly what the container does on each poll.

To run without the network, supply scans from a directory. The project writes
synthetic ones through its own decoder, so they cannot drift from the real
format:

```bash
uv run python tools/make_sample_inputs.py \
    --into data/raw --run-time 2025-08-28T22:00:00Z
uv run irene-nowcast --config config/default.json \
    --source local --input-dir data/raw --output-dir outputs --device cpu
```

A directory may hold either format, one file per scan on the full 1 km DPC
grid, in mm/h: `dpc_sri.YYYYMMDDHHMM.tif` as the API serves it (`-9999` outside
coverage), or `dpc_sri.YYYYMMDDHHMM.nc` with a variable `RR`, as the sample
published beside the model.

A run writes `outputs/irene/dpc_sri/YYYY/MM/DD/HHMMZ/` holding 48 GeoTIFFs with
the default configuration (12 leads × 4 members) and a `manifest.json`
recording the inputs, artifacts, stages, versions, device, seed and checksums
that produced them.

## Configuration

`config/default.json` carries the runtime's keys plus two of IRENE's own:

| Key | Default | Meaning |
| --- | --- | --- |
| `remote_directory` | `SRI` | The API's name for the product: surface rain intensity. |
| `ftp` | 3 attempts | Timeout and retry schedule for the API. The key is the runtime's name for it; nothing in it is FTP. |
| `domain` | 256 × 256 km | Sub-rectangle of the DPC grid, in metres. Any size works; one that is not a multiple of 32 cells is padded and cropped back. |
| `max_missing_fraction` | `0.95` | Radar coverage leaves about half the rectangle empty, so the 0.05 that suits a satellite product would reject every scan. |
| `missing_policy` | `zero_rain` | Gaps are fed to the model as no rain and masked as NoData in the output. |
| `ensemble_members` | `4` | Members per lead. The model was published with 10. |
| `seed` | `20260101` | Members come from fresh noise, so an unseeded run is not reproducible. `null` leaves it unseeded. |

## What is not settled

- **The licence and limits of the feed.** See above; settle them before
  depending on this service.
- **Only CPU is exercised.** `--device cuda` is accepted and the image carries
  CPU-only torch.
- **No skill is claimed.** Conformance checks the contracts: units, grid,
  window spacing, output shape, checksums. Whether IRENE has skill on the data
  you give it is a question for its authors' validation and your own.
