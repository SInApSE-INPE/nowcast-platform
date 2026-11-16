"""The two formats the DPC rain-rate mosaic is published in.

Neither is a built-in, because neither is a raw block, and both follow the
ordinary `Decoder` contract.

`RadarGeoTiff` is the operational format: what the DPC open-data API serves,
one LZW GeoTIFF per five-minute scan. It carries its own projection and
transform, so this decoder *checks* them against the declared grid instead of
trusting either side. Cells outside radar coverage hold -9999.

`RadarNetcdf` is the format of the sample published beside the model. It is
NetCDF4, which is HDF5 underneath, so h5py reads it without the netCDF C
library. Two things about that sample are worth stating rather than silently
working around, because both are the kind of mismatch that produces plausible
output and no error:

- The ``RR`` variable declares ``units = kg m-2 s-1``. Its values do not agree:
  a maximum of 61.6 with a mean of 0.61 and 84% zeros is a rain rate in mm/h,
  while the same figure in kg m-2 s-1 would be 220,000 mm/h. The model card
  says mm/h. This decoder reports mm/h and ignores the attribute.
- The variable carries ``grid_mapping = "crs"``, and the sample contains no
  such variable, so the NetCDF alone does not say where it is. The projection
  is known from the GeoTIFF of the same mosaic, on the same 1200 x 1400 grid.

`EitherFormat` lets one deployment read both, so a directory of downloaded
scans and the published sample are replayed by the same command.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from nowcast_runtime.builtin.decoders import TimestampedName
from nowcast_runtime.ports import Frame, Grid


@dataclass(frozen=True)
class RadarNetcdf:
    """One rain-rate field per file, on a projected, north-up metre grid."""

    native_grid: Grid
    naming: TimestampedName
    variable: str = "RR"
    units: str = "mm/h"
    suffix: str = ".nc"

    def __post_init__(self):
        if not self.units:
            raise ValueError("A decoder must declare physical units")

    @property
    def glob(self) -> str:
        return "*" + self.suffix

    def observed_at(self, name: str) -> datetime | None:
        return self.naming.observed_at(name)

    def filename(self, observed: datetime) -> str:
        return self.naming.filename(observed)

    def read(self, path: Path, grid: Grid) -> Frame:
        observed = self.observed_at(path.name)
        if observed is None:
            raise ValueError(f"Not an observation this decoder recognizes: {path.name}")
        row, column = self.native_grid.locate(grid)
        with h5py.File(path, "r") as handle:
            if self.variable not in handle:
                raise ValueError(f"{path.name} has no {self.variable} variable")
            field = handle[self.variable]
            if field.ndim != 2:
                raise ValueError(
                    f"{path.name} holds a {field.ndim}-dimensional {self.variable}; "
                    "this decoder reads one observation per file"
                )
            if field.shape != (self.native_grid.height, self.native_grid.width):
                raise ValueError(
                    f"{path.name} is {field.shape}, not the declared native grid "
                    f"{(self.native_grid.height, self.native_grid.width)}"
                )
            block = field[row : row + grid.height, column : column + grid.width]
        values = np.asarray(block, dtype=np.float32)
        # Radar coverage does not fill the rectangle, so a large NaN fraction is
        # ordinary here rather than a fault. The mask carries it onward.
        missing = ~np.isfinite(values)
        return Frame(values[np.newaxis], missing[np.newaxis], self.units, grid, observed)

    def synthesize(self, path: Path, values: np.ndarray, grid: Grid) -> Path:
        """Write a file this decoder reads back as values on grid.

        The inverse of `read`, so a conformance suite can exercise the whole
        deployment without any real radar data.
        """
        if values.shape != grid.shape:
            raise ValueError("Synthetic values must match the requested grid")
        row, column = self.native_grid.locate(grid)
        field = np.full(
            (self.native_grid.height, self.native_grid.width), np.nan, dtype=np.float32
        )
        field[row : row + grid.height, column : column + grid.width] = values
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset(self.variable, data=field, dtype="float32")
            dataset.attrs["units"] = self.units
            dataset.attrs["long_name"] = "Total precipitation rate"
        return path


@dataclass(frozen=True)
class RadarGeoTiff:
    """One rain-rate field per GeoTIFF, as the DPC open-data API serves it."""

    native_grid: Grid
    naming: TimestampedName
    units: str = "mm/h"
    # What the provider writes outside radar coverage. The file declares no
    # NoData value, so it has to be named here.
    nodata: float = -9999.0
    suffix: str = ".tif"

    def __post_init__(self):
        if not self.units:
            raise ValueError("A decoder must declare physical units")

    @property
    def glob(self) -> str:
        return "*" + self.suffix

    def observed_at(self, name: str) -> datetime | None:
        return self.naming.observed_at(name)

    def filename(self, observed: datetime) -> str:
        return self.naming.filename(observed)

    def _transform(self):
        native = self.native_grid
        return from_origin(native.west, native.north, native.resolution_x, native.resolution_y)

    def _open(self, path: Path):
        """Open a scan, refusing one that is not on the declared grid.

        The file states its own projection and transform. Comparing them with
        the declared grid is what turns the CRS from an assumption into a fact
        checked on every scan: a provider that regrids fails here, loudly,
        instead of being published a few kilometres off.
        """
        try:
            dataset = rasterio.open(path)
        except rasterio.errors.RasterioIOError as exc:
            raise ValueError(f"{path.name} is not a readable GeoTIFF") from exc
        native = self.native_grid
        problem = None
        if dataset.count != 1:
            problem = f"holds {dataset.count} bands; this decoder reads one field per file"
        elif (dataset.height, dataset.width) != (native.height, native.width):
            problem = (
                f"is {(dataset.height, dataset.width)}, not the declared native grid "
                f"{(native.height, native.width)}"
            )
        elif not dataset.transform.almost_equals(self._transform()):
            problem = "is not on the declared native grid origin and resolution"
        elif dataset.crs is None or dataset.crs != CRS.from_string(native.crs):
            problem = f"declares a projection other than {native.crs}"
        if problem:
            dataset.close()
            raise ValueError(f"{path.name} {problem}")
        return dataset

    def validate(self, path: Path) -> None:
        """Decode the whole scan, so a truncated transfer is caught on arrival."""
        with self._open(path) as dataset:
            try:
                dataset.read(1)
            except rasterio.errors.RasterioIOError as exc:
                raise ValueError(f"{path.name} is truncated or corrupt") from exc

    def read(self, path: Path, grid: Grid) -> Frame:
        observed = self.observed_at(path.name)
        if observed is None:
            raise ValueError(f"Not an observation this decoder recognizes: {path.name}")
        row, column = self.native_grid.locate(grid)
        with self._open(path) as dataset:
            try:
                block = dataset.read(1, window=Window(column, row, grid.width, grid.height))
            except rasterio.errors.RasterioIOError as exc:
                raise ValueError(f"{path.name} is truncated or corrupt") from exc
        values = np.asarray(block, dtype=np.float32)
        # Out of coverage is a sentinel in this format. It becomes NaN here so
        # no later stage can mistake -9999 for a rain rate and clip it to dry.
        missing = ~np.isfinite(values) | (values == self.nodata)
        values = np.where(missing, np.nan, values).astype(np.float32)
        return Frame(values[np.newaxis], missing[np.newaxis], self.units, grid, observed)

    def synthesize(self, path: Path, values: np.ndarray, grid: Grid) -> Path:
        """Write a file this decoder reads back as values on grid."""
        if values.shape != grid.shape:
            raise ValueError("Synthetic values must match the requested grid")
        native = self.native_grid
        row, column = native.locate(grid)
        field = np.full((native.height, native.width), self.nodata, dtype=np.float32)
        field[row : row + grid.height, column : column + grid.width] = np.where(
            np.isfinite(values), values, self.nodata
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=native.height,
            width=native.width,
            count=1,
            dtype="float32",
            crs=native.crs,
            transform=self._transform(),
            compress="lzw",
        ) as dataset:
            dataset.write(field, 1)
        return path


@dataclass(frozen=True)
class EitherFormat:
    """Several formats of one product, told apart by file name.

    The first is the primary one: it names and writes synthetic scans. All of
    them must describe the same grid in the same units, since they are the same
    observations in different containers.
    """

    formats: tuple

    def __post_init__(self):
        if not self.formats:
            raise ValueError("At least one format is required")
        first = self.formats[0]
        for other in self.formats[1:]:
            if other.native_grid != first.native_grid or other.units != first.units:
                raise ValueError("Formats of one product must share a grid and units")

    @property
    def native_grid(self) -> Grid:
        return self.formats[0].native_grid

    @property
    def units(self) -> str:
        return self.formats[0].units

    @property
    def glob(self) -> str:
        return "*"

    def _recognizing(self, name: str):
        for candidate in self.formats:
            if candidate.observed_at(name) is not None:
                return candidate
        return None

    def observed_at(self, name: str) -> datetime | None:
        candidate = self._recognizing(name)
        return None if candidate is None else candidate.observed_at(name)

    def filename(self, observed: datetime) -> str:
        return self.formats[0].filename(observed)

    def read(self, path: Path, grid: Grid) -> Frame:
        candidate = self._recognizing(path.name)
        if candidate is None:
            raise ValueError(f"Not an observation this decoder recognizes: {path.name}")
        return candidate.read(path, grid)

    def synthesize(self, path: Path, values: np.ndarray, grid: Grid) -> Path:
        return self.formats[0].synthesize(path, values, grid)
