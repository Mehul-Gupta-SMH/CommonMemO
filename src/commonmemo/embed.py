"""An opt-in local embedder (#129, ADR 0005): `pip install "commonmemo[embed]"`.

    from commonmemo import Memory
    from commonmemo.embed import local_ranker

    mem = Memory(ranker=local_ranker())      # keyword + meaning, on this machine, no API

The model is `minishlab/potion-base-32M` (Model2Vec static embeddings, about 120 MB, 0.3 ms a window).
It is pinned by name, revision and the SHA-256 of its weights: weights that do not match fail, and
so does a missing network or cache. There is no silent fallback to keyword ranking. To work offline,
download the model once with network (`huggingface-cli download minishlab/potion-base-32M --revision
<REVISION>`) and pass `local_ranker(path=<that folder>)`, or set `HF_HOME` to a cache that has it.
English-only: the model is trained on English text. `pip install commonmemo` alone needs none of this
(the module imports `model2vec` only when `local_ranker` runs).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from .ranking import HybridRanker, sync_embedder

MODEL = "minishlab/potion-base-32M"
REVISION = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
"""The commit of `MODEL` the checksum is for. Pinned by `.github/workflows/embed-pin.yml`, which prints both."""
SHA256 = "99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28"
"""SHA-256 of `model.safetensors` at `REVISION`."""


class EmbedUnavailable(RuntimeError):
    """The embedder cannot be made: the extra is missing, the weights are, or they are not the pinned ones."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise EmbedUnavailable('the embedder needs the extra: pip install "commonmemo[embed]"') from e
    try:
        return Path(snapshot_download(MODEL, revision=revision))
    except Exception as e:  # no network, no cache, unknown revision
        raise EmbedUnavailable(
            f"cannot get {MODEL}@{revision} ({type(e).__name__}); download it once with network "
            f"(huggingface-cli download {MODEL} --revision {revision}) and pass local_ranker(path=<folder>)") from e


def load(path: Optional[str] = None, *, revision: str = REVISION, sha256: str = SHA256, model: Any = None) -> Any:
    """The Model2Vec model, checked against the pinned checksum. `model` is for tests: it skips loading."""
    if model is not None:
        return model
    if not (revision and sha256):
        raise EmbedUnavailable("the model revision and checksum are not pinned in this build; refusing to load unverified weights")
    try:
        from model2vec import StaticModel
    except ImportError as e:
        raise EmbedUnavailable('the embedder needs the extra: pip install "commonmemo[embed]"') from e
    folder = Path(path) if path else _fetch(revision)
    weights = folder / "model.safetensors"
    if not weights.is_file():
        raise EmbedUnavailable(f"{weights} not found: not a {MODEL} folder")
    if _sha256(weights) != sha256:
        raise EmbedUnavailable(f"{weights} does not match the pinned checksum for {MODEL}@{revision}")
    return StaticModel.from_pretrained(str(folder))


def local_ranker(path: Optional[str] = None, *, threshold: float = 0.1, model: Any = None) -> HybridRanker:
    """Keyword and meaning ranking with the local model, for `Memory(ranker=...)`. Raises `EmbedUnavailable`."""
    m = load(path, model=model)
    return HybridRanker(sync_embedder(lambda text: m.encode([text])[0].tolist()), threshold=threshold)


__all__ = ["EmbedUnavailable", "MODEL", "REVISION", "SHA256", "load", "local_ranker"]
