"""
Embedding backends for trial graph chunk index & node retrieval (Step A/D).

Default: MiniMax REST embeddings (model emb-o01, 1536-d), cosine similarity after L2 normalize.

Fallback: sentence-transformers (see API.md).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None  # type: ignore


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def cosine_top1(query_vec: np.ndarray, matrix_rows: np.ndarray) -> int:
    """query (dim,), matrix (n, dim) — both L2-normalized internally."""
    q = query_vec.astype(np.float32).reshape(-1)
    q /= np.linalg.norm(q) + 1e-12
    m = _normalize_rows(matrix_rows)
    sims = m @ q
    return int(np.argmax(sims))


def _coerce_vector_list(raw: Any) -> list[list[float]]:
    """Normalize list[vector] from API (float lists or {embedding: [...]} dicts)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("empty or non-list vectors")
    out: list[list[float]] = []
    for item in raw:
        if isinstance(item, list) and item and isinstance(item[0], (int, float)):
            out.append([float(x) for x in item])
        elif isinstance(item, dict):
            emb = item.get("embedding") or item.get("vector") or item.get("values")
            if isinstance(emb, list):
                out.append([float(x) for x in emb])
            else:
                raise ValueError(f"dict item missing embedding: {list(item.keys())}")
        else:
            raise ValueError(f"unexpected vector item type: {type(item)}")
    return out


def _parse_vectors(payload: dict[str, Any]) -> list[list[float]]:
    """Accept multiple MiniMax / proxy response shapes."""
    base_resp = payload.get("base_resp")
    if isinstance(base_resp, dict):
        status = base_resp.get("status_code") or base_resp.get("status")
        if status not in (None, 0, "0", 200, "success", "Success"):
            msg = base_resp.get("status_msg") or base_resp.get("message") or base_resp
            raise ValueError(f"Embedding API error in base_resp: {msg}")

    for key in ("vectors", "embeddings"):
        if key in payload:
            return _coerce_vector_list(payload[key])

    if "data" in payload:
        data = payload["data"]
        if isinstance(data, list) and data:
            if isinstance(data[0], dict) and "embedding" in data[0]:
                ordered = sorted(data, key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in ordered if isinstance(d, dict)]
            return _coerce_vector_list(data)

    if isinstance(base_resp, dict):
        for key in ("vectors", "embeddings", "embedding", "data"):
            if key in base_resp:
                try:
                    inner = base_resp[key]
                    if isinstance(inner, list) and inner and isinstance(inner[0], (int, float)):
                        return [[float(x) for x in inner]]
                    return _coerce_vector_list(inner)
                except ValueError:
                    continue
    raise ValueError(f"Unrecognized embedding response keys: {list(payload.keys())}")


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")  # type: ignore[attr-defined]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[attr-defined]
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding HTTP {e.code}: {detail}") from e
    return json.loads(raw)


def embed_minimax_rest(texts: list[str], embed_type: str) -> np.ndarray:
    """
    MiniMax-style embeddings HTTP. Tries payloads with optional `type` field.
    embed_type: 'db' for corpus chunks, 'query' for retrieval queries.
    """
    if urllib is None:
        raise RuntimeError("urllib unavailable")

    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY or ANTHROPIC_API_KEY required for MiniMax embeddings.")

    base = os.environ.get("MINIMAX_HTTP_BASE", "https://api.minimaxi.com").rstrip("/")
    url = os.environ.get("MINIMAX_EMBEDDING_URL", f"{base}/v1/embeddings")
    model = os.environ.get("MINIMAX_EMBEDDING_MODEL", "emb-o01")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    gid = os.environ.get("MINIMAX_GROUP_ID")
    if gid:
        headers["Group-Id"] = gid

    attempts: list[dict[str, Any]] = [
        {"model": model, "texts": texts, "type": embed_type},
        {"model": model, "texts": texts},
        {"model": model, "input": texts},
    ]

    last_err: Exception | None = None
    for body in attempts:
        try:
            payload = _post_json(url, headers, body)
            vecs = _parse_vectors(payload)
            if len(vecs) != len(texts):
                raise ValueError(f"Expected {len(texts)} vectors, got {len(vecs)}")
            return np.asarray(vecs, dtype=np.float32)
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"MiniMax embedding failed after retries: {last_err}")


_st_model_instance = None
_st_model_id: str | None = None


def _sentence_transformer_singleton():
    """Load ST model once per process (avoid reloading on every embed_batch call)."""
    global _st_model_instance, _st_model_id
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "pip install sentence-transformers torch for EMBEDDING_BACKEND=sentence_transformers"
        ) from e

    mid = os.environ.get(
        "ST_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    if _st_model_instance is None or _st_model_id != mid:
        _st_model_instance = SentenceTransformer(mid)
        _st_model_id = mid
    return _st_model_instance


def embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    model = _sentence_transformer_singleton()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def embed_batch(texts: list[str], embed_type: str) -> np.ndarray:
    backend = os.environ.get("EMBEDDING_BACKEND", "minimax_rest").strip().lower()
    if backend == "sentence_transformers":
        return embed_sentence_transformers(texts)
    # Default: MiniMax HTTP in mini-batches (provider limits vary)
    batch = int(os.environ.get("MINIMAX_EMBED_BATCH", "16"))
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        sub = texts[i : i + batch]
        parts.append(embed_minimax_rest(sub, embed_type))
    return np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)


__all__ = ["embed_batch", "cosine_top1"]
