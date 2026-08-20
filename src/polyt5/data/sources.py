"""Declarative registry of external datasets and a provenance-recording downloader.

The polyT5 authors' corpora are withheld (see ``docs/data.md``), so this
repository runs on public substitutes. Every substitute is declared here --
URL, size, license, citation -- and fetched by :func:`download`, which writes
a JSON sidecar (``<file>.provenance.json``) recording the URL, byte size and
SHA-256 of what was fetched, so a corpus version can be pinned and verified.

Importing this module has no side effects: no network access, no file writes.

Policy: sources flagged ``requires_confirmation`` (the 9 GB polyOne corpus)
are refused unless the caller passes ``confirm_large=True`` (the CLI flag
``--yes-large``), and the refusal states the size.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "SOURCES",
    "ConfirmationRequiredError",
    "DataSource",
    "DownloadRecord",
    "download",
]

_CHUNK_BYTES = 1 << 20  # 1 MiB read chunks for streaming download / hashing.


class ConfirmationRequiredError(RuntimeError):
    """Raised when a large download is requested without explicit confirmation."""


@dataclass(frozen=True)
class DataSource:
    """A single external dataset: where it lives and under what terms.

    Attributes:
        name: Registry key, e.g. ``"pi1m"``.
        url: Direct download URL.
        filename: Local filename to store under the destination directory.
        description: One-line human summary.
        license: License string (verbatim from the source).
        approx_bytes: Approximate download size, for user-facing reporting and
            for the large-download refusal message. Not used for verification.
        requires_confirmation: If True, :func:`download` refuses to fetch over
            the network unless ``confirm_large=True``.
        citation: Reference to cite when the data is used.
    """

    name: str
    url: str
    filename: str
    description: str
    license: str
    approx_bytes: int
    requires_confirmation: bool
    citation: str


@dataclass
class DownloadRecord:
    """Provenance of one fetched (or adopted) file, serialized as a sidecar.

    Attributes:
        name: Registry key of the source.
        url: URL the file was (or would be) fetched from.
        path: Absolute path of the local file.
        bytes: Exact on-disk byte size.
        sha256: Hex SHA-256 digest of the file contents.
        downloaded_utc: ISO-8601 UTC timestamp of the fetch/adoption.
        license: License string carried over from the source.
    """

    name: str
    url: str
    path: str
    bytes: int
    sha256: str
    downloaded_utc: str
    license: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of this record."""
        return asdict(self)


SOURCES: dict[str, DataSource] = {
    "pi1m": DataSource(
        name="pi1m",
        url="https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M_v2.csv",
        filename="PI1M_v2.csv",
        description="PI1M v2: ~1M generated star-terminated PSMILES with SA scores",
        license="MIT (repository; README notes academic use)",
        approx_bytes=66_000_000,
        requires_confirmation=False,
        citation=(
            "Ma & Luo, PI1M: A Benchmark Database for Polymer Informatics, "
            "J. Chem. Inf. Model. 60, 4684-4690 (2020)"
        ),
    ),
    "lamalab_tg": DataSource(
        name="lamalab_tg",
        url=(
            "https://zenodo.org/records/15210035/files/"
            "LAMALAB_CURATED_Tg_structured_polymerclass.csv?download=1"
        ),
        filename="LAMALAB_CURATED_Tg.csv",
        description="LamaLab curated experimental polymer Tg dataset (7,367 rows)",
        license="CC-BY-4.0",
        approx_bytes=5_300_000,
        requires_confirmation=False,
        citation=(
            "LamaLab curated Tg dataset, Zenodo record 15210035 "
            "(doi:10.5281/zenodo.15210035); see also the polymetrix package"
        ),
    ),
    "polyone_train": DataSource(
        name="polyone_train",
        url=(
            "https://zenodo.org/records/7766806/files/"
            "generated_polymer_smiles_train.txt?download=1"
        ),
        filename="generated_polymer_smiles_train.txt",
        description="polyOne/polyBERT corpus, train portion: 80M generated PSMILES",
        license="Other (Non-Commercial) -- GTRC academic-use license",
        approx_bytes=8_140_000_000,
        requires_confirmation=True,
        citation=(
            "Kuenneth & Ramprasad, polyBERT, Nat. Commun. 14, 4099 (2023); "
            "Zenodo record 7766806 (doi:10.5281/zenodo.7766806)"
        ),
    ),
    "polyone_dev": DataSource(
        name="polyone_dev",
        url=(
            "https://zenodo.org/records/7766806/files/"
            "generated_polymer_smiles_dev.txt?download=1"
        ),
        filename="generated_polymer_smiles_dev.txt",
        description="polyOne/polyBERT corpus, dev portion: 20M generated PSMILES",
        license="Other (Non-Commercial) -- GTRC academic-use license",
        approx_bytes=900_000_000,
        requires_confirmation=True,
        citation=(
            "Kuenneth & Ramprasad, polyBERT, Nat. Commun. 14, 4099 (2023); "
            "Zenodo record 7766806 (doi:10.5281/zenodo.7766806)"
        ),
    ),
}


def _sha256_of(path: Path) -> str:
    """Stream-hash a file with SHA-256 and return the hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(dest: Path) -> Path:
    """Return the provenance sidecar path for a data file."""
    return dest.with_name(dest.name + ".provenance.json")


def _write_sidecar(record: DownloadRecord, dest: Path) -> None:
    """Write the provenance record next to the data file."""
    _sidecar_path(dest).write_text(
        json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def _record_for_existing(source: DataSource, dest: Path) -> DownloadRecord:
    """Build (and persist) a provenance record for a file already on disk.

    Fast path: when a sidecar exists and its recorded byte size matches the
    file, the stored record is returned without re-hashing. Otherwise the file
    is hashed and a fresh sidecar is written -- still with no network access.

    Args:
        source: The registry entry the file belongs to.
        dest: Path of the existing file.

    Returns:
        The provenance record for the on-disk file.
    """
    size = dest.stat().st_size
    sidecar = _sidecar_path(dest)
    if sidecar.exists():
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
        if stored.get("bytes") == size and stored.get("sha256"):
            return DownloadRecord(**stored)
    record = DownloadRecord(
        name=source.name,
        url=source.url,
        path=str(dest.resolve()),
        bytes=size,
        sha256=_sha256_of(dest),
        downloaded_utc=datetime.now(timezone.utc).isoformat(),
        license=source.license,
    )
    _write_sidecar(record, dest)
    return record


def download(
    source: DataSource,
    dest_dir: Path,
    *,
    confirm_large: bool = False,
    force: bool = False,
) -> DownloadRecord:
    """Fetch a registered dataset, or adopt an already-present copy.

    Behavior:
        * If ``dest_dir/source.filename`` already exists and ``force`` is
          False, no network access happens: the file is adopted (hashed, and a
          provenance sidecar is written if missing or stale). Confirmation is
          not required for adoption -- it gates network transfers only.
        * A source with ``requires_confirmation=True`` is refused with
          :class:`ConfirmationRequiredError` (stating the size) unless
          ``confirm_large=True``.
        * Downloads stream to a ``.part`` file and are moved into place only
          when complete, then hashed, and the sidecar is written.

    Args:
        source: Registry entry to fetch.
        dest_dir: Directory to place the file in (created if missing).
        confirm_large: Explicit approval for ``requires_confirmation`` sources
            (the CLI's ``--yes-large``).
        force: Re-download even if the file already exists.

    Returns:
        The :class:`DownloadRecord` for the on-disk file.

    Raises:
        ConfirmationRequiredError: A large source was requested over the
            network without ``confirm_large=True``.
        urllib.error.URLError: The transfer failed.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename

    if dest.exists() and not force:
        return _record_for_existing(source, dest)

    if source.requires_confirmation and not confirm_large:
        approx_gb = source.approx_bytes / 1e9
        raise ConfirmationRequiredError(
            f"{source.name!r} is ~{approx_gb:.2f} GB ({source.approx_bytes} bytes); "
            f"license: {source.license}. Pass confirm_large=True (CLI: --yes-large) "
            f"to download it."
        )

    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": "polyt5-rlvr/0.1"})
    with urllib.request.urlopen(request) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, _CHUNK_BYTES)
    part.replace(dest)

    record = DownloadRecord(
        name=source.name,
        url=source.url,
        path=str(dest.resolve()),
        bytes=dest.stat().st_size,
        sha256=_sha256_of(dest),
        downloaded_utc=datetime.now(timezone.utc).isoformat(),
        license=source.license,
    )
    _write_sidecar(record, dest)
    return record
