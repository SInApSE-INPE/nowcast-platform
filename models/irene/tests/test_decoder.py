"""The two formats of the DPC mosaic, and the checks the GeoTIFF one makes."""

from datetime import UTC, datetime

import numpy as np
import pytest
import rasterio
from irene_nowcast.entrypoint import NATIVE_GRID, decoder, geotiff, netcdf
from rasterio.transform import from_origin

from nowcast_runtime.ports import Grid

OBSERVED = datetime(2026, 11, 16, 22, 35, tzinfo=UTC)
GRID = Grid(NATIVE_GRID.crs, -200000.0, 256000.0, 1000.0, 1000.0, 64, 32)


def field():
    values = np.linspace(0, 12, GRID.width * GRID.height, dtype=np.float32).reshape(GRID.shape)
    values[:4, :] = np.nan
    return values


@pytest.mark.parametrize("make", [geotiff, netcdf])
def test_each_format_reads_back_what_it_wrote(tmp_path, make):
    reader = make()
    path = reader.synthesize(tmp_path / reader.filename(OBSERVED), field(), GRID)
    frame = decoder().read(path, GRID)
    assert frame.time == OBSERVED and frame.units == "mm/h" and frame.grid == GRID
    assert frame.missing[0, :4].all() and not frame.missing[0, 4:].any()
    np.testing.assert_array_equal(frame.values[0, 4:], field()[4:])


def test_out_of_coverage_is_missing_not_a_negative_rain_rate(tmp_path):
    reader = geotiff()
    path = reader.synthesize(tmp_path / reader.filename(OBSERVED), field(), GRID)
    with rasterio.open(path) as dataset:
        assert (dataset.read(1) == -9999).any(), "the file holds the provider sentinel"
    frame = reader.read(path, NATIVE_GRID)
    assert np.isnan(frame.values[frame.missing]).all()
    assert (frame.values[~frame.missing] >= 0).all()


def test_the_declared_projection_is_the_one_the_dpc_publishes():
    crs = rasterio.crs.CRS.from_string(NATIVE_GRID.crs)
    published = rasterio.crs.CRS.from_wkt(
        'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
        '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",42],'
        'PARAMETER["central_meridian",12.5],PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
    )
    assert crs == published
    assert crs != rasterio.crs.CRS.from_epsg(3003)


def write(path, crs=NATIVE_GRID.crs, west=NATIVE_GRID.west, shape=None, count=1):
    height, width = shape or (NATIVE_GRID.height, NATIVE_GRID.width)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=crs, transform=from_origin(west, NATIVE_GRID.north, 1000.0, 1000.0),
    ) as dataset:  # fmt: skip
        dataset.write(np.zeros((count, height, width), dtype=np.float32))
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"crs": "EPSG:3003"}, "projection"),
        ({"west": NATIVE_GRID.west + 1000.0}, "origin"),
        ({"shape": (700, 600)}, "native grid"),
        ({"count": 2}, "2 bands"),
    ],
)
def test_a_scan_that_is_not_on_the_declared_grid_is_refused(tmp_path, change, message):
    path = write(tmp_path / geotiff().filename(OBSERVED), **change)
    with pytest.raises(ValueError, match=message):
        geotiff().validate(path)
    with pytest.raises(ValueError, match=message):
        geotiff().read(path, GRID)


def test_a_truncated_scan_is_refused(tmp_path):
    reader = geotiff()
    path = reader.synthesize(tmp_path / reader.filename(OBSERVED), field(), GRID)
    path.write_bytes(path.read_bytes()[:-2000])
    with pytest.raises(ValueError, match="truncated|readable"):
        reader.validate(path)


def test_names_decide_the_format_and_strangers_are_ignored():
    reader = decoder()
    assert reader.observed_at("dpc_sri.202611162235.tif") == OBSERVED
    assert reader.observed_at("dpc_sri.202611162235.nc") == OBSERVED
    assert reader.observed_at("manifest.json") is None
    assert reader.filename(OBSERVED).endswith(".tif")
