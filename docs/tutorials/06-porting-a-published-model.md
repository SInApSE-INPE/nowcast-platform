
# Porting a published model: IRENE

This is the platform's own acceptance test, written up as a walkthrough.

[IRENE](https://huggingface.co/it4lia/irene) is a ConvGRU ensemble nowcaster
from Fondazione Bruno Kessler, trained on the Italian DPC radar mosaic and
published on HuggingFace under BSD 2-Clause. It was chosen because it is
unlike the model this runtime grew around on every axis that matters:

| | InceptionV4 | IRENE |
| --- | --- | --- |
| Time step | 1 hour | **5 minutes** |
| Window | 12 observations | 6 observations |
| Horizon | 6 leads | **12 leads** |
| Output | deterministic | **ensemble, 10 members** |
| Architecture | direct encoder-decoder | **recurrent ConvGRU** |
| Source | GSMaP binary over FTP | **radar GeoTIFF over an HTTP API** |
| Grid | 0.1° geographic | **1 km projected, in metres** |
| Transform | log1p and z-score | Marshall-Palmer dBZ |
| Bundle | checkpoint, mean, std | checkpoint only |

```{admonition} The result, stated plainly
:class: important

IRENE runs, passes all nineteen conformance checks, and now operates as a
container that publishes 48 georeferenced ensemble forecasts for every new
radar scan. Getting there required **two changes to the runtime** and **two to
the conformance suite**, and it exposed **one mistake of the port's own**: a
projection that was asserted, wrong, and published for a while. All five are
described below, because a platform that only reports its successes is not
worth trusting.
```

![Observed Italian radar against the IRENE ensemble mean at three lead times](../_static/irene-ensemble.png)

Real Italian radar, 29 August 2025. The top row is what the radar recorded;
the bottom row is the ensemble mean forecast for the same moments. Detail
decays with lead time, which is what a nowcaster does. The white region in
the +30 min observation is a coverage dropout in that scan, carried through
as NoData rather than invented.

```{warning}
One case is an **illustration, not a verification**. No score is computed
here and none should be inferred. Whether IRENE has skill over Italy is a
question for its authors and its own validation record.
```

## 1. Read the model before writing anything

The facts that shaped every later decision, from the model card and the source:

- `predict(past, forecast_steps, ensemble_size)` takes `(T, H, W)` float32 in
  mm/h and returns `(E, T, H, W)` in mm/h.
- **The normalization lives inside `predict`**: a Marshall-Palmer dBZ transform,
  `Z = 200 R^1.6`, clipped to 0-60 dBZ and mapped to [-1, 1]. No mean, no std.
- The published checkpoint is a **770 MB PyTorch Lightning artefact** needing
  `weights_only=False`, because its hyperparameters pickle optimizer classes.
- A plain `nn.Module`, `EncoderDecoder`, sits underneath and needs only torch.
- Fully convolutional, but both dimensions must be a multiple of 32.

## 2. Scaffold, then replace

**From the workspace root:**

```bash
uv run nowcast-new-model --name "IRENE" --slug irene \
    --product dpc_sri --units "mm/h" --into models/irene
```

Everything that follows is inside `models/irene/`. The files this port
touched, beyond what the scaffold wrote:

```text
models/irene/
├── config/default.json              edited: domain, policy, members, seed
├── artifacts/v1/                    replaced: real bundle and manifest
├── src/irene_nowcast/
│   ├── entrypoint.py                rewritten: composition root
│   ├── model.py                     rewritten: predictor and transform
│   ├── decoder.py                   new: the two DPC formats
│   ├── source.py                    new: the DPC open-data API
│   ├── weights.py                   new: shrink the checkpoint
│   ├── stages.py                    new: the Threshold postprocess
│   └── _vendor/                     new: upstream architecture, verbatim
│       ├── LICENSE
│       ├── model.py
│       └── utils.py
├── tests/test_conformance.py        edited: realistic radar fields
└── tools/prepare_artifacts.py       new: the same, as a script
```

## 3. Shrink the checkpoint before it reaches production

Two thirds of the published file is optimizer state, schedulers and loop state.
Keeping it means every deployment downloads and checksums 770 MB to run a
257 MB model, and loads it with `weights_only=False` on every container start.

`models/irene/src/irene_nowcast/weights.py` does that conversion **once per
bundle**, never on the inference path. It is in the package, with
`tools/prepare_artifacts.py` as a thin script over it, because the container
runs it on start to fill a bundle it was not built with. Making a new bundle
from a checkpoint you already hold, from `models/irene/`:

```bash
uv run irene-download-artifacts \
    --checkpoint /path/to/model.ckpt --bundle artifacts/v2
```

```text
Extracted 40 tensors
     770,192,441 bytes  model.ckpt
     256,722,959 bytes  irene.pt  (33.3%)
```

The resulting bundle holds tensors only, so the operational loader reads it
with `weights_only=True`. The manifest records the digest of the source
checkpoint, so the provenance chain back to the Hub is unbroken.

That chain is also how anyone else gets the weights, which are not in Git.
Given a bundle that already has its manifest, the same tool downloads the
checkpoint the manifest names, refuses it unless the digest matches, and keeps
the extracted weights only if they match too:

```bash
uv run irene-download-artifacts --bundle artifacts/v1
```

The digest pinned for the weights is the one the locked torch produces. A
different torch build can serialize the same tensors to different bytes, and
the tool then stops with both digests rather than keep a file the manifest
does not describe. The container builds from the same lock, which is why it
reproduces the digest.

```{tip}
Do this for any Lightning checkpoint. The unpickling that a training artefact
requires is a code-execution path, and confining it to one step, on a file
whose digest is pinned, is much better than repeating it on every run.
```

## 4. Vendor only what you need

The published package requires Python 3.13 and declares twenty-odd
dependencies including JupyterLab, TensorBoard and pysteps. Inference needs
none of that.

`models/irene/src/irene_nowcast/_vendor/` holds `model.py` and `utils.py`
copied verbatim, with the upstream BSD 2-Clause licence beside them. Both
import only torch and numpy.

```{important}
Vendored code is kept **byte-identical** to upstream so it can still be diffed
against it, which means excluding it from the linter rather than reformatting
it. The workspace does that with `extend-exclude = ["**/_vendor/**"]`.
```

## 5. Declare the transform rather than hiding it

The published `predict` normalizes internally. Declaring it as the plugin's
`Transform` instead means the runtime applies it, and **it appears in every run
manifest**:

```{code-block} json
:caption: from a run manifest, models/irene/outputs/.../manifest.json

"normalization": {
  "kind": "marshall-palmer-dbz",
  "relation": "Z = 200 * R**1.6",
  "decibel_range": [0, 60],
  "representable_mm_h": [0.0364633, 205.048],
  "clamped_before_inverse": true
}
```

Two properties of that scale matter operationally, and neither is obvious:

**It saturates at the bottom.** Model space `-1` inverts to 0.0365 mm/h, not
zero. Without intervention every dry pixel in the domain carries a uniform
drizzle that no forecast intended.

**The published inference path omits the clamp used in training.** A member
drifting past `+1` grows exponentially through the inverse: `1.2` becomes
486 mm/h, `1.5` becomes 1776.

Both are handled by a `Threshold` postprocess step, which is exactly what the
postprocess port is for.

```{warning}
Derive bounds like these **from the transform**, not from a literal.

Writing `OUTPUT_FLOOR = 0.036463` looked right and was wrong by less than a
float32 ulp, so `values <= floor` matched nothing and the first real run came
out with a uniform 0.0365 mm/h over the whole domain and zero exact zeros. It
looked like a forecast. Computing the floor as
`normalized_to_rainrate(-1.0)` makes the comparison exact by construction.
```

## 6. Write a decoder for the real format

The sample published beside the model is NetCDF4, which is HDF5 underneath, so
`h5py` reads it without the netCDF C library. The decoder is about ninety
lines and implements `native_grid`, `units`, `observed_at`, `read`, plus the
optional `filename` and `synthesize` that let conformance run on no real data.

The operational feed turned out to be GeoTIFF, not NetCDF. That second decoder
came later and is described in [section 9](#going-operational).

Two things in the published sample are worth recording rather than silently
accommodating:

```{admonition} The units attribute is wrong
:class: warning

`RR` declares `units = kg m-2 s-1`. Its values — maximum 61.6, mean 0.61, 84%
zeros, no negatives — are a rain rate in **mm/h**. The same figure in
kg m⁻² s⁻¹ would be 220,000 mm/h. Every code path in the upstream package
treats it as mm/h. Applying the 3600× conversion the attribute implies would
produce numbers that are catastrophically wrong and perfectly plausible to a
pipeline that does not check.
```

```{admonition} The projection is not in the sample, and the first guess was wrong
:class: warning

The variable carries `grid_mapping = "crs"` and the file contains no such
variable. The port first **asserted** `EPSG:3003` (Monte Mario / Italy zone 1),
declared in one place and recorded in every manifest so it could be
contradicted. It was contradicted: the GeoTIFFs the DPC API serves, on the
same 1200 × 1400 grid with the same origin, state a transverse Mercator
centred on 42°N 12.5°E on WGS 84, which has no EPSG code. Every forecast
written before that was georeferenced in the wrong place, and nothing failed.

In `models/irene/src/irene_nowcast/entrypoint.py`:

    DPC_CRS = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    NATIVE_GRID = Grid(DPC_CRS, -600000.0, 650000.0, 1000.0, 1000.0, 1200, 1400)

The lesson is the reason the assertion was written down at all: a CRS nobody
checked is a guess, and it stays one until a file that carries the answer says
otherwise. The GeoTIFF decoder now compares each scan's own projection and
transform with this grid and refuses a scan that disagrees.

Coordinates in the file are cell centres, so the grid origin is half a cell
beyond: x starts at -599500 and the origin is -600000.
```

## 7. Configure the deployment

```{code-block} json
:caption: models/irene/config/default.json

{
  "domain": { "west": -200000.0, "south": 0.0, "east": 56000.0, "north": 256000.0 },
  "product": "dpc_sri",
  "missing_policy": "zero_rain",
  "max_missing_fraction": 0.95,
  "ensemble_members": 4,
  "seed": 20260101
}
```

Three choices worth explaining:

**`max_missing_fraction: 0.95`.** Radar coverage fills about half of the
rectangle, so more than half of a real scan is NaN. The 0.05 that suits a
satellite product would reject every observation.

**`missing_policy: zero_rain`, not `normalized_zero`.** For this transform the
value that maps to model-space zero is about 2.7 mm/h — moderate rain, not an
absence of it. Filling gaps with it would invent precipitation over everywhere
the radar cannot see.

**`seed`.** Members come from fresh noise per decoder pass, so an unseeded run
is not reproducible. The seed is recorded in the manifest.

The domain is 256 × 256 cells, which satisfies the model's multiple-of-32
constraint without padding.

## 8. Run it

The project generates synthetic scans for a smoke test, which is what the
conformance suite also uses. **From the workspace root:**

```bash
uv run --directory models/irene python tools/make_sample_inputs.py \
    --into data/raw --run-time 2025-08-28T22:00:00Z
uv run --directory models/irene irene-nowcast \
    --config config/default.json --source local \
    --input-dir data/raw --output-dir outputs --device cpu
```

### On real radar

The figure above is not from those. The repository's published sample,
`examples/sample_data.nc`, holds 54 real scans of 29 August 2025, and the
run shown used the six wettest consecutive ones:

```text
  dpc_sri.202508290015.nc  nan  1.5%  wet  10667  max  90.19 mm/h
  dpc_sri.202508290020.nc  nan  1.5%  wet  11099  max  71.06 mm/h
  dpc_sri.202508290025.nc  nan  1.5%  wet  10675  max  66.56 mm/h
  dpc_sri.202508290030.nc  nan  1.5%  wet  10305  max  73.00 mm/h
  dpc_sri.202508290035.nc  nan  1.5%  wet  11004  max  78.12 mm/h
  dpc_sri.202508290040.nc  nan  1.5%  wet  11165  max  74.19 mm/h
```

What came out, against 11,165 wet pixels observed at the run time with a
peak of 74.19 mm/h:

```text
  + 5 min : wet  22853  max  54.03  mean 2.143 mm/h
  +15 min : wet  15549  max  62.23  mean 3.211 mm/h
  +30 min : wet  15816  max  33.39  mean 2.586 mm/h
  +60 min : wet  15333  max  23.12  mean 1.767 mm/h
```

Structure persists and intensity decays, which is the expected behaviour.
Ensemble spread at +30 min is 0.35 mm/h against 0.03 on the synthetic
input: real weather gives the decoder more to be uncertain about.

```text
INFO Model ready on cpu; 6 input steps
INFO Acquired 6 consecutive inputs ending at 2025-08-28T22:00:00+00:00
INFO Generated lead +1: 2025-08-28T22:05:00+00:00
...
INFO Generated lead +12: 2025-08-28T23:00:00+00:00
INFO Published complete run: outputs/irene/dpc_sri/2025/08/28/2200Z
```

48 files, 12 leads × 4 members:

```text
irene_20250828T2200Z_f001_m000_20250828T2205Z.tif
irene_20250828T2200Z_f001_m001_20250828T2205Z.tif
...
```

And conformance, which at this point was the project's whole test suite:

```text
...................                                                      [100%]
19 passed in 62.09s
```

(going-operational)=
## 9. Going operational

The DPC serves the mosaic through an open API, with no account: one call names
the newest product time, another exchanges a product type and a time for a
short-lived pre-signed URL. That needed three things, all inside
`models/irene/`:

- **A source**, `source.py`. There is no directory to list, so it answers
  "what exists" by asking for exactly the times the window needs. A time the
  API does not have is left out, and the runtime then waits (`WAITING`)
  instead of forecasting from a shorter history. The pre-signed URL carries
  temporary credentials, so it stays in memory: the location recorded in the
  manifest is the product type and time.
- **A second decoder**, `RadarGeoTiff`, for what the API serves: one LZW
  GeoTIFF per scan, with `-9999` outside radar coverage. Left alone,
  `ClipNegative` would have turned that sentinel into "no rain"; the decoder
  reports it as missing instead. `EitherFormat` lets the same deployment read
  both formats, told apart by file name.
- **The bundle on start.** `irene-serve` runs the extraction from section 3
  before the first poll, so an image holds the manifest and no weights.

### A source without a listing

A `Source` answers two questions: what exists, and give me this one. FTP
answers the first with a directory listing. This API has none, so `candidates`
asks about the window directly:

```{code-block} python
:caption: models/irene/src/irene_nowcast/source.py (abridged)

def candidates(self, window, run_time):
    anchor = run_time if run_time is not None else self.latest()
    if anchor is None:
        return []
    entries = []
    for moment in window.times(anchor):
        name = self._filename(moment)
        if (self._directory / name).is_file() or self._locate(moment) is not None:
            entries.append((name, self._location(moment)))
    return entries
```

Nothing else was needed for the scheduler to work. Its probe is a window of
one observation, so the same method tells it whether anything new exists, and
`select_window` already turns an absent time into `DataNotReady`. The port did
not have to grow a "latest" method.

`fetch` follows the built-in FTP source: download to a `.part` file, validate,
rename into a cache keyed by source identity. Validation is the decoder's
`validate`, which decodes the whole raster, so a truncated transfer is retried
instead of cached.

### The second decoder checks what the first one assumed

```{code-block} python
:caption: models/irene/src/irene_nowcast/decoder.py (abridged)

elif not dataset.transform.almost_equals(self._transform()):
    problem = "is not on the declared native grid origin and resolution"
elif dataset.crs is None or dataset.crs != CRS.from_string(native.crs):
    problem = f"declares a projection other than {native.crs}"
```

The wrong `EPSG:3003` of section 6 was found by reading the header of the
first downloaded scan by hand. With this check in place the same mistake is
refused at the first scan instead. A format that carries its own
georeferencing should always be compared with the declared grid, because
either side can be the one that is wrong.

### Configuration

Two keys of the runtime's own are reused rather than duplicated:

```{code-block} json
:caption: models/irene/config/default.json

"remote_directory": "SRI",
"ftp": { "timeout_seconds": 60, "attempts": 3,
         "retry_delay_seconds": 30, "max_retry_delay_seconds": 120 }
```

`remote_directory` holds the API's name for the product, and `ftp` is the
retry schedule, which has nothing FTP-specific in it. Both names predate any
remote but FTP; renaming them is a breaking change this port did not justify.

### Run it

From `models/irene/`:

```bash
mkdir -p outputs && docker compose up -d --build
docker compose logs -f
```

```text
INFO Starting IRENE with a 60-second poll interval
irene.pt: already verified
INFO NEW_DATA: processing 2026-11-16T22:50:00+00:00
INFO Model ready on cpu; 6 input steps
INFO Acquired 6 consecutive inputs ending at 2026-11-16T22:50:00+00:00
INFO Published complete run: outputs/irene/dpc_sri/2026/11/16/2250Z
INFO UP_TO_DATE: 2026-11-16T22:50:00+00:00 already completed
```

A scan appears about seven minutes after its nominal time, the service asks
once a minute, and a run on four CPU threads takes under twenty seconds. The
same poll runs by hand, without Docker, as
`uv run irene-nowcast --config config/default.json --source dpc --scheduled`.

### Test it without the network

The source takes its transport as a parameter, so its tests script the API's
answers: a missing scan, an expired URL, a truncated transfer, a server error.
None of them opens a socket. With the decoder tests, the project's suite is
now 42 tests, 19 of them the shared conformance checks.

```{warning}
The documentation found for this API states neither a licence nor a rate
limit. That is a question to settle with the provider before depending on the
feed, and no amount of engineering on this side answers it. The
[IRENE README](https://github.com/SInApSE-INPE/nowcast-platform/tree/main/models/irene)
lists this beside the other operational notes: who owns the output files, and
that nothing rotates them.
```

## What this cost the platform

### One runtime change, for the port

`Config.load` called `cls(**payload)`, so **any key the runtime did not own was
a TypeError**. IRENE needs two settings of its own, and the only alternatives
were a second configuration file per deployment or model settings on the
command line, which `pipeline.main` does not admit either.

`Config` now carries unknown keys in `extra`. Additive, defaulted, no plugin
changes: runtime 0.3.0 → 0.3.1.

### A second one, on going operational

`--source` accepted `local` or `ftp`, and a scheduled poll required `ftp` by
name. IRENE's remote is an HTTP API, and calling it `ftp` to get past an
argument check would have put a false protocol in every command line.
`pipeline.parser` and `pipeline.main` now take the deployment's own `sources`,
and the scheduler refuses only `local`. Additive and defaulted again: runtime
0.3.1 → 0.3.2.

### Two conformance suite changes

**The suite built a frame no decoder could produce.** `raw_frame` paired values
containing gaps with an all-empty mask, and a gap-filling stage then failed on
it. The mask is now derived from the values, as every decoder derives it.

**The suite assumed transforms are invertible.** A reflectivity scale is
deliberately lossy, so `inverse(forward(0))` is the floor rather than zero. A
transform can now declare `representable_minimum` and `representable_maximum`,
and the round-trip check narrows to the range it claims.

### Nothing else

The ports, the sink, the stages, the orchestration and the atomic publication
needed no change at all. The five-minute step, the non-uniform
window machinery, the ensemble member axis and the declared output width all
worked the first time they met a model that actually used them.

## What this does not show

```{warning}
**No skill is demonstrated here.** One case, no scores, no baseline. The
forecast looks reasonable beside the observation, and that is an anecdote.

The CRS is now read from the provider's own files, after a first assertion
that was wrong. Whether IRENE is useful over Italy is a
question for its authors and its own validation record, and nothing in this
platform will answer it.
```

### A limitation the port actually hit

The published sample is a **datacube**: 54 timesteps in one NetCDF. The
contracts assume **one observation per object** — `observed_at` maps one file
name to one time, and `fetch` returns a `Path` with nowhere to carry an index.

This was not hypothetical. Running on the real sample required splitting the
cube into per-scan files first, outside the pipeline, because there is no way
to express it inside.

The operational DPC feed publishes one file per five-minute scan, so there it
never matters. For a provider that publishes cubes, this is the first
genuinely missing port, and it would need either a source that explodes a cube
into virtual observations or a `fetch` that can carry an index.
