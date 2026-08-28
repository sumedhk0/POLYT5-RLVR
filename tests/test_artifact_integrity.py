"""The checked-in artifacts must still hash to what frozen_baseline.json recorded.

scripts/train_grpo.py verifies each baseline artifact's SHA-256 before training and
REFUSES to run on a mismatch. Git's end-of-line conversion silently breaks that for
text artifacts: the tokenizer was committed such that a Windows checkout produced
CRLF (hashing 48983573..., the recorded value) while a Linux checkout produced LF
(hashing 691030e2...), so a fresh clone on Linux could not train at all.

.gitattributes now marks artifacts/** as -text. This test is the guard: it fails on
any platform whose checkout does not reproduce the recorded bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN = REPO_ROOT / "artifacts" / "baseline" / "frozen_baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_the_tokenizer_still_hashes_to_its_frozen_record():
    """The one sha256-verified artifact that is text, and so the only one at risk.

    The .pt checkpoints are binary; git does not convert those. This file is JSON,
    and it is the file whose hash mismatch blocked training on a fresh Linux clone.
    """
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    recorded = frozen["artifacts"]["tokenizer"]["sha256"]
    path = REPO_ROOT / "artifacts" / "tokenizer" / "polyt5_vocab.json"
    actual = _sha256(path)
    assert actual == recorded, (
        f"{path} hashes to {actual} but frozen_baseline.json records {recorded}. "
        "If the only difference is line endings, .gitattributes is not being honoured "
        "for artifacts/** -- train_grpo.py will refuse to train against this checkout."
    )


def test_every_tracked_frozen_artifact_that_exists_matches_its_record():
    """Checkpoints are gitignored, so skip the ones a clean clone will not have."""
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    checked = 0
    for key, meta in frozen["artifacts"].items():
        path = REPO_ROOT / str(meta["path"]).replace("\\", "/")
        if not path.is_file():
            continue
        checked += 1
        assert _sha256(path) == meta["sha256"], f"{key} at {path} no longer matches"
    if checked == 0:
        pytest.skip("no frozen artifacts present in this checkout")
