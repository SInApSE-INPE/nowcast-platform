"""The composition root: the only file knowing both the model and the adapters."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from datetime import timedelta

from nowcast_runtime import pipeline, service
from nowcast_runtime.artifacts import verify_bundle
from nowcast_runtime.builtin.decoders import TimestampedName
from nowcast_runtime.builtin.sinks import GeoTiffSink
from nowcast_runtime.builtin.sources import LocalFiles
from nowcast_runtime.builtin.stages import ClipNegative, MissingPolicy
from nowcast_runtime.composition import Deployment
from nowcast_runtime.config import Config
from nowcast_runtime.ports import Grid, ModelPlugin

from . import __version__
from .decoder import EitherFormat, RadarGeoTiff, RadarNetcdf
from .model import OUTPUT_CEILING, OUTPUT_FLOOR, library_versions, load_model, transform
from .source import DpcRadarApi
from .stages import Threshold

NAME = "IRENE"
SLUG = "irene"
CONFIG = "config/default.json"
BUNDLE = "artifacts/v1"
# `dpc` is the DPC open-data API; there is no FTP feed for this product.
SOURCES = ("local", "dpc")
# Said at every start: the model is someone else's work, and this is a test.
NOTICE = (
    "IRENE is an existing model by Fondazione Bruno Kessler, published at "
    "https://huggingface.co/it4lia/irene. This deployment is a test of the "
    "nowcast-platform workflow, not an official distribution of the model."
)
# A scan is published about seven minutes after its nominal time and replaced
# by the next one five minutes later, so a window an hour old means the feed
# has stalled, and forecasting from it would be issuing the past.
POLL_ARGS = [
    "--config", CONFIG,
    "--source", "dpc",
    "--scheduled",
    "--output-dir", "outputs",
    "--max-age-hours", "1",
]  # fmt: skip
# How often the container asks whether a new scan exists. Scans arrive every
# five minutes; asking every minute bounds the delay between a scan appearing
# and its forecast starting, and an unchanged answer costs one small request.
POLL_INTERVAL_SECONDS = 60

# The projection the DPC mosaic is published in: a transverse Mercator centred
# on Italy, on WGS 84, with no false origin. It has no EPSG code, so it is
# written out. This is read from the GeoTIFFs the DPC API serves, and
# `RadarGeoTiff` re-checks it against every scan it opens.
DPC_CRS = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# 1 km cells, rows north first, 1200 x 1400. The NetCDF sample gives cell
# centres, so its first x is -599500 for this outer corner of -600000.
NATIVE_GRID = Grid(DPC_CRS, -600000.0, 650000.0, 1000.0, 1000.0, 1200, 1400)

PLUGIN = ModelPlugin(
    name=NAME,
    slug=SLUG,
    version=__version__,
    bundle_roles=("weights",),
    requires_runtime=">=0.3,<0.4",
    load=load_model,
    transform=transform,
    # Both ends of the reflectivity scale saturate, and the published inference
    # path applies neither bound. Zeroing the floor keeps dry pixels dry instead
    # of a uniform 0.0365 mm/h drizzle over the whole domain.
    postprocess=lambda artifacts: (
        Threshold(floor=OUTPUT_FLOOR, ceiling=OUTPUT_CEILING),
    ),
    library_versions=library_versions,
)


def geotiff() -> RadarGeoTiff:
    """One GeoTIFF per five-minute scan, as the DPC API serves it."""
    return RadarGeoTiff(
        native_grid=NATIVE_GRID,
        naming=TimestampedName(
            pattern=r"dpc_sri\.(?P<stamp>\d{12})\.tif",
            stamp_format="%Y%m%d%H%M",
            template="dpc_sri.%Y%m%d%H%M.tif",
        ),
        units="mm/h",
    )


def netcdf() -> RadarNetcdf:
    """One NetCDF per scan, variable RR: the sample published with the model."""
    return RadarNetcdf(
        native_grid=NATIVE_GRID,
        naming=TimestampedName(
            pattern=r"dpc_sri\.(?P<stamp>\d{12})\.nc",
            stamp_format="%Y%m%d%H%M",
            template="dpc_sri.%Y%m%d%H%M.nc",
        ),
        variable="RR",
        units="mm/h",
    )


def decoder() -> EitherFormat:
    """Both formats of the mosaic; the operational GeoTIFF is the primary one."""
    return EitherFormat((geotiff(), netcdf()))


def compose(args: argparse.Namespace) -> tuple[Deployment, str]:
    config = Config.load(args.config)
    reader = decoder()
    grid = config.domain.to_grid(reader.native_grid)

    if args.source == "local":
        if args.input_dir is None:
            raise ValueError("--source local requires --input-dir")
        source = LocalFiles(args.input_dir, reader.glob)
    else:
        operational = geotiff()
        source = DpcRadarApi(
            # The API's name for the product: SRI, the surface rain intensity.
            product_type=config.remote_directory,
            cache=args.cache_dir,
            # The retry schedule is the runtime's; its key predates any source
            # but FTP, and nothing in it is specific to that protocol.
            retry=config.ftp,
            filename=operational.filename,
            verify=operational.validate,
        )

    bundle, artifacts = verify_bundle(config.artifacts, PLUGIN.bundle_roles)

    # Ensemble size and the noise seed are operationally variable, so they live
    # in configuration. The runtime carries keys it does not own in `extra`.
    members = int(config.extra.get("ensemble_members", 10))
    seed = config.extra.get("seed")
    seed = None if seed is None else int(seed)
    plugin = replace(
        PLUGIN,
        load=lambda a, device, g: load_model(a, device, g, members=members, seed=seed),
        library_versions=lambda: {
            **library_versions(),
            "irene.ensemble_members": str(members),
            "irene.seed": "unseeded" if seed is None else str(seed),
        },
    )

    deployment = Deployment(
        plugin=plugin,
        source=source,
        decoder=reader,
        sink=GeoTiffSink(variable_names=("precipitation_rate",)),
        grid=grid,
        bundle=bundle,
        artifacts=artifacts,
        device=args.device,
        preprocess=(
            # A negative rain rate would become NaN under R**1.6.
            ClipNegative(),
            # Radar coverage leaves most of the rectangle empty, so the limit
            # here is far looser than a satellite product would need. The mask
            # carries the gaps to the output, which the published inference path
            # does not do.
            MissingPolicy(config.missing_policy, config.max_missing_fraction),
        ),
    )
    return deployment, config.product


def parser():
    return pipeline.parser(SLUG, NAME, SOURCES)


def run(args):
    deployment, product = compose(args)
    return pipeline.run(args, deployment, product)


def main(argv=None):
    return pipeline.main(compose, SLUG, NAME, argv, SOURCES)


def serve():
    """Fill the bundle, then forecast from every new scan until told to stop."""
    from . import weights

    # Configured here, in the format the service uses, so the notice is the
    # first line of the container log rather than lost before logging exists.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).info(NOTICE)
    # A default, not an override: the environment still decides when it is set.
    os.environ.setdefault(service.POLL_INTERVAL_ENV, str(POLL_INTERVAL_SECONDS))
    return service.serve(
        NAME,
        download=lambda: weights.main(["--bundle", BUNDLE]),
        poll=lambda: main(POLL_ARGS),
    )


__all__ = ["NAME", "SLUG", "PLUGIN", "compose", "decoder", "parser", "run", "main", "serve"]

# Spacing between scans, exported so a sample generator can use it.
SCAN_INTERVAL = timedelta(minutes=5)
