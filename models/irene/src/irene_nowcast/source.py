"""The DPC radar open-data API as a `Source`.

The Dipartimento della Protezione Civile serves its national mosaic through two
calls, with no account and no key:

- ``GET /findLastProductByType?type=SRI`` names the newest product time;
- ``POST /downloadProduct`` exchanges a product type and a time for a
  pre-signed S3 URL that is good for a few minutes.

There is no listing to read, so this source answers "what exists" by asking for
exactly the times a window needs. A time the API does not have is left out, and
the runtime then waits for it instead of forecasting from a shorter history.

Pre-signed URLs carry temporary credentials. They are held in memory for the
one transfer and never reach a log line or a run manifest: an observation's
recorded location is its product type and time, which is all that identifies it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nowcast_runtime.builtin.sources import PARTIAL, IncompleteDownload, RetryPolicy
from nowcast_runtime.ports import Observation, WindowSpec

LOG = logging.getLogger(__name__)
API = "https://radar-api.protezionecivile.it"
# The API rejects calls that do not name the DPC viewer as their origin.
ORIGIN = "https://radar.protezionecivile.it"
# A URL is dropped this long before it expires, so a transfer never starts on
# one that lapses halfway.
EXPIRY_MARGIN_SECONDS = 60.0

# method, url, headers, body, timeout -> status, payload
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def http(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    """One HTTP exchange. A refusal is a status to act on, not an exception."""
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


class DpcRadarApi:
    """One DPC product, fetched scan by scan into a verified local cache."""

    def __init__(
        self,
        product_type: str,
        cache: Path,
        retry: RetryPolicy,
        filename: Callable[[datetime], str],
        verify: Callable[[Path], None],
        base_url: str = API,
        transport: Transport = http,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not product_type.isidentifier():
            raise ValueError(f"Not a DPC product type: {product_type!r}")
        self.product_type = product_type
        self.cache = cache
        self.retry = retry
        self.base_url = base_url.rstrip("/")
        self._filename = filename
        self._verify = verify
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        # time -> (pre-signed URL, monotonic deadline). Memory only, on purpose.
        self._urls: dict[datetime, tuple[str, float]] = {}

    @property
    def identity(self) -> str:
        return f"{self.base_url}/{self.product_type}"

    @property
    def live(self) -> bool:
        """The newest scan keeps moving, so an old one is a stalled feed."""
        return True

    @property
    def _directory(self) -> Path:
        # Namespaced by API and product, so two products never share a name.
        return self.cache / hashlib.sha256(self.identity.encode()).hexdigest()[:16]

    def _exchange(self, method: str, url: str, body: Mapping[str, object] | None = None):
        """Call the API, retrying what may pass and returning what will not.

        A 404 is an answer ("no such product yet"), so it is returned. A network
        failure, a throttle or a server error is retried on the configured
        schedule. Nothing from the server is echoed into the error raised.
        """
        headers = {"origin": ORIGIN, "accept": "application/json"}
        payload = None
        if body is not None:
            headers["content-type"] = "application/json"
            payload = json.dumps(body).encode()
        for attempt in range(self.retry.attempts):
            try:
                status, content = self._transport(
                    method, url, headers, payload, self.retry.timeout_seconds
                )
            except OSError:
                status, content = None, b""
            if status in (200, 404):
                return status, content
            if attempt + 1 == self.retry.attempts:
                raise ConnectionError(
                    f"DPC radar API request failed after {self.retry.attempts} attempts "
                    f"(last status: {status or 'no response'})"
                )
            delay = self.retry.delay(attempt)
            LOG.warning(
                "DPC radar API request failed; retry %s/%s in %ss",
                attempt + 2,
                self.retry.attempts,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable: attempts is always positive")

    def latest(self) -> datetime | None:
        """The newest product time the API reports, or None when it has none."""
        query = urlencode({"type": self.product_type})
        status, content = self._exchange("GET", f"{self.base_url}/findLastProductByType?{query}")
        if status == 404:
            return None
        products = json.loads(content).get("lastProducts") or []
        if not products:
            return None
        return datetime.fromtimestamp(products[0]["time"] / 1000, UTC)

    def _locate(self, moment: datetime) -> str | None:
        """A usable pre-signed URL for one scan, or None when it does not exist."""
        held = self._urls.get(moment)
        if held is not None and self._clock() < held[1]:
            return held[0]
        self._urls.pop(moment, None)
        status, content = self._exchange(
            "POST",
            f"{self.base_url}/downloadProduct",
            {"productType": self.product_type, "productDate": int(moment.timestamp() * 1000)},
        )
        if status == 404:
            return None
        answer = json.loads(content)
        lifetime = float(answer.get("expiresSeconds", 0)) - EXPIRY_MARGIN_SECONDS
        self._urls[moment] = (answer["url"], self._clock() + lifetime)
        return answer["url"]

    def _location(self, moment: datetime) -> str:
        return f"{self.identity}/{moment:%Y-%m-%dT%H:%MZ}"

    def candidates(
        self, window: WindowSpec, run_time: datetime | None
    ) -> Sequence[tuple[str, str]]:
        """Each scan of the window that is on disk or that the API can serve."""
        anchor = run_time if run_time is not None else self.latest()
        if anchor is None:
            return []
        entries = []
        for moment in window.times(anchor):
            name = self._filename(moment)
            if (self._directory / name).is_file() or self._locate(moment) is not None:
                entries.append((name, self._location(moment)))
        return entries

    def fetch(self, observation: Observation) -> Path:
        target = self._directory / observation.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                self._verify(target)
                return target
            except (OSError, EOFError, ValueError):
                LOG.warning("Invalid cached file; downloading %s again", observation.name)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=PARTIAL, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            self._transfer(observation, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _transfer(self, observation: Observation, temporary: Path) -> None:
        for attempt in range(self.retry.attempts):
            url = self._locate(observation.time)
            if url is None:
                raise FileNotFoundError(f"The DPC API no longer serves {observation.name}")
            try:
                status, content = self._transport(
                    "GET", url, {}, None, self.retry.timeout_seconds
                )
                if status != 200:
                    raise IncompleteDownload(f"Transfer refused with status {status}")
                temporary.write_bytes(content)
                try:
                    self._verify(temporary)
                except (EOFError, ValueError) as exc:
                    raise IncompleteDownload(str(exc)) from exc
                return
            except OSError:
                # Whatever went wrong, the URL is not trusted for a second try.
                self._urls.pop(observation.time, None)
                if attempt + 1 == self.retry.attempts:
                    # The URL is a credential: name the scan, never the address.
                    raise ConnectionError(
                        f"Could not download {observation.name} "
                        f"after {self.retry.attempts} attempts"
                    ) from None
                delay = self.retry.delay(attempt)
                LOG.warning(
                    "Transfer of %s failed; retry %s/%s in %ss",
                    observation.name,
                    attempt + 2,
                    self.retry.attempts,
                    delay,
                )
                self._sleep(delay)

    def prune(self, ttl: timedelta, now: float | None = None) -> int:
        """Remove scans older than ttl from this source's own cache directory."""
        if ttl <= timedelta(0):
            raise ValueError("cache TTL must be positive")
        directory = self._directory
        if directory.is_symlink() or not directory.is_dir():
            return 0
        cutoff = (time.time() if now is None else now) - ttl.total_seconds()
        removed = 0
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        return removed
