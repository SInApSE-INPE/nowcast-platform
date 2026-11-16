"""The DPC API source, driven by a scripted transport: no network is touched."""

import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from irene_nowcast.source import DpcRadarApi

from nowcast_runtime.acquisition import DataNotReady, catalog, select_window
from nowcast_runtime.builtin.decoders import TimestampedName
from nowcast_runtime.builtin.sources import RetryPolicy
from nowcast_runtime.ports import Observation, WindowSpec

LATEST = datetime(2026, 11, 16, 22, 35, tzinfo=UTC)
WINDOW = WindowSpec.uniform(timedelta(minutes=5), 6)
NAMING = TimestampedName(
    pattern=r"dpc_sri\.(?P<stamp>\d{12})\.tif",
    stamp_format="%Y%m%d%H%M",
    template="dpc_sri.%Y%m%d%H%M.tif",
)
SECRET = "X-Amz-Signature=do-not-log-me"


class Api:
    """A stand-in for the two DPC endpoints and the bucket behind them."""

    def __init__(self, latest=LATEST, absent=(), expires=900):
        self.latest, self.absent, self.expires = latest, set(absent), expires
        self.calls, self.failures, self.body = [], [], b"scan"

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url.split("?")[0].rsplit("/", 1)[-1]))
        if self.failures:
            outcome = self.failures.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome, b"{}"
        if "findLastProductByType" in url:
            assert headers["origin"] == "https://radar.protezionecivile.it"
            if self.latest is None:
                return 404, b'{"total":0,"lastProducts":[]}'
            stamp = int(self.latest.timestamp() * 1000)
            return 200, json.dumps({"lastProducts": [{"time": stamp}]}).encode()
        if url.endswith("/downloadProduct"):
            request = json.loads(body)
            assert request["productType"] == "SRI"
            moment = datetime.fromtimestamp(request["productDate"] / 1000, UTC)
            if moment in self.absent:
                return 404, b'{"error":"File non trovato"}'
            address = f"https://bucket.example/{request['productDate']}.tif?{SECRET}"
            return 200, json.dumps({"url": address, "expiresSeconds": self.expires}).encode()
        assert SECRET in url, "a transfer must use the pre-signed address"
        return 200, self.body

    def count(self, endpoint):
        return sum(1 for _, called in self.calls if called == endpoint)


def source(tmp_path, api, verify=lambda path: None, clock=lambda: 0.0, attempts=3):
    return DpcRadarApi(
        product_type="SRI",
        cache=tmp_path / "cache",
        retry=RetryPolicy(5, attempts, 1, 4),
        filename=NAMING.filename,
        verify=verify,
        transport=api,
        sleep=lambda seconds: None,
        clock=clock,
    )


def observation(moment=LATEST):
    return Observation(moment, NAMING.filename(moment), "SRI")


def test_the_latest_scan_anchors_a_window_when_no_time_is_given(tmp_path):
    api = Api()
    found = catalog(source(tmp_path, api), NAMING, WINDOW, None)
    assert [item.time for item in select_window(found, WINDOW)] == list(WINDOW.times(LATEST))
    assert api.count("findLastProductByType") == 1


def test_an_explicit_time_never_asks_for_the_latest(tmp_path):
    api = Api()
    catalog(source(tmp_path, api), NAMING, WINDOW, LATEST - timedelta(hours=3))
    assert api.count("findLastProductByType") == 0


def test_a_scan_the_api_lacks_makes_the_window_wait(tmp_path):
    gap = LATEST - timedelta(minutes=10)
    found = catalog(source(tmp_path, Api(absent=[gap])), NAMING, WINDOW, None)
    with pytest.raises(DataNotReady, match=re.escape(gap.isoformat())):
        select_window(found, WINDOW)


def test_no_product_at_all_is_waiting_not_failure(tmp_path):
    found = catalog(source(tmp_path, Api(latest=None)), NAMING, WINDOW, None)
    with pytest.raises(DataNotReady):
        select_window(found, WINDOW)


def test_the_presigned_address_never_reaches_a_recorded_location(tmp_path):
    for item in catalog(source(tmp_path, Api()), NAMING, WINDOW, None):
        assert "Signature" not in item.location and "bucket.example" not in item.location
        assert item.location.endswith(f"/SRI/{item.time:%Y-%m-%dT%H:%MZ}")


def test_a_cached_scan_is_neither_asked_about_nor_downloaded_again(tmp_path):
    api = Api()
    feed = source(tmp_path, api)
    first = feed.fetch(observation())
    assert first.read_bytes() == b"scan"
    api.calls.clear()
    catalog(feed, NAMING, WindowSpec((timedelta(0),)), LATEST)
    assert feed.fetch(observation()) == first
    assert api.calls == []


def test_an_address_found_while_listing_is_reused_for_the_transfer(tmp_path):
    api = Api()
    feed = source(tmp_path, api)
    catalog(feed, NAMING, WINDOW, None)
    feed.fetch(observation())
    assert api.count("downloadProduct") == len(WINDOW)


def test_an_expired_address_is_requested_again(tmp_path):
    now = [0.0]
    api = Api(expires=900)
    feed = source(tmp_path, api, clock=lambda: now[0])
    catalog(feed, NAMING, WindowSpec((timedelta(0),)), LATEST)
    now[0] = 900.0
    feed.fetch(observation())
    assert api.count("downloadProduct") == 2


def test_a_truncated_transfer_is_retried_and_never_cached(tmp_path):
    api = Api()
    seen = []

    def verify(path):
        seen.append(path.read_bytes())
        if len(seen) == 1:
            raise ValueError("truncated")

    feed = source(tmp_path, api, verify=verify)
    path = feed.fetch(observation())
    assert len(seen) == 2 and path.name == observation().name
    assert [item.name for item in path.parent.iterdir()] == [path.name]


def test_server_errors_are_retried_then_reported_without_the_address(tmp_path):
    api = Api()
    api.failures = [503, OSError("reset"), 429]
    with pytest.raises(ConnectionError) as failure:
        source(tmp_path, api).latest()
    assert "3 attempts" in str(failure.value)

    api = Api()
    feed = source(tmp_path, api, verify=lambda path: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ConnectionError) as failure:
        feed.fetch(observation())
    assert SECRET not in str(failure.value) and "bucket.example" not in str(failure.value)
    assert not list((tmp_path / "cache").rglob("*.tif"))
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_a_scan_withdrawn_before_transfer_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="dpc_sri.202611162235.tif"):
        source(tmp_path, Api(absent=[LATEST])).fetch(observation())


def test_prune_removes_only_expired_scans_of_this_source(tmp_path):
    feed = source(tmp_path, Api())
    kept = feed.fetch(observation())
    old = feed.fetch(observation(LATEST - timedelta(minutes=5)))
    bystander = tmp_path / "cache" / "not-mine.tif"
    bystander.write_bytes(b"x")
    now = old.stat().st_mtime
    import os

    os.utime(old, (now - 7200, now - 7200))
    os.utime(bystander, (now - 7200, now - 7200))
    assert feed.prune(timedelta(hours=1), now=now) == 1
    assert kept.exists() and not old.exists() and bystander.exists()


def test_a_product_type_is_one_word(tmp_path):
    with pytest.raises(ValueError, match="product type"):
        DpcRadarApi("SRI/../x", tmp_path, RetryPolicy(5, 1, 1, 1), NAMING.filename, print)
