#!/usr/bin/env python3
"""
Build session-level knowledge graph from a trial_profile JSON (RAG bundle).

Goal: feed downstream LLMs that write **eligibility criteria**. The graph carries
*criteria-useful knowledge concepts* — not case identifiers (no NCT / PMID nodes).

Pipeline:
  A) Fine-grained chunks (short & specific) + embedding index.
  B) LLM in **batches of N chunks** (`GRAPH_CHUNK_BATCH_SIZE`, default 10): each round receives
     **only** that batch's chunks; model outputs plain-text lines
     `*E1 (phrase)* relation *E2 (phrase)* <Extracted|Inferred>`.
  C) Code parses lines, merges nodes (stable ids), synonym-collapses relations, sets edge
     `frequency`, aggregates `kind` (any extracted → extracted).
  D) Per-node retrieval query (name + category + 1-hop edges) → embed → Top-1 chunk_ref.

A node is defined by the set of its edges (no description; relations alone reveal meaning).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from collections import Counter

from trial_graph.embeddings import cosine_top1, embed_batch

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore


MAX_CHUNK_CHARS = 420  # soft upper bound
MIN_CHUNK_CHARS = 120  # below this we merge a chunk into the previous one (same source block)

ENTITY_SPAN = r"\*([^*]+?)\s*\(([^)]+)\)\*"
ENTITY_SPAN_PLAIN = r"\*([^*]+?)\*"
EVIDENCE_TAG = r"<\s*(Extracted|Inferred)\s*>"
LINE_RE = re.compile(
    rf"^\s*{ENTITY_SPAN}\s+(.+?)\s+{ENTITY_SPAN}\s*{EVIDENCE_TAG}\s*$",
    re.IGNORECASE,
)
LINE_RE_PLAIN = re.compile(
    rf"^\s*{ENTITY_SPAN_PLAIN}\s+(.+?)\s+{ENTITY_SPAN_PLAIN}\s*{EVIDENCE_TAG}\s*$",
    re.IGNORECASE,
)

RELATION_SYNONYM_GROUPS: dict[str, frozenset[str]] = {
    "treated_by": frozenset({
        "treated by", "treatment with", "therapy with", "administered", "administered by",
        "is treated by", "receives treatment with",
    }),
    "indicated_for": frozenset({
        "indicated for", "indication for", "used for", "approved for", "is indicated for",
    }),
    "targets": frozenset({
        "targets", "inhibits", "blocks", "antagonizes", "targets protein",
    }),
    "associated_with": frozenset({
        "associated with", "linked to", "related to", "is associated with",
    }),
    "causes": frozenset({
        "causes", "leads to", "results in", "can cause",
    }),
    "contraindicated_in": frozenset({
        "contraindicated in", "contraindication for", "is contraindicated in",
    }),
}


def _node_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


@dataclass
class ParsedEdge:
    src_name: str
    src_def: str
    relation: str
    dst_name: str
    dst_def: str
    kind: str  # extracted | inferred


@dataclass
class ParseStats:
    ok: int = 0
    fallback_ok: int = 0
    skipped_empty: int = 0
    missing_tag: int = 0
    bad_format: int = 0
    sample_failures: list[str] | None = None

    def __post_init__(self) -> None:
        if self.sample_failures is None:
            self.sample_failures = []


def normalize_relation(rel: str) -> str:
    s = re.sub(r"\s+", " ", rel.strip().lower())
    s = s.strip(".,;:!?\"'")
    s = re.sub(r"^(?:is|are|was|were)\s+", "", s)
    return s


def _build_norm_to_canon_from_groups(groups: dict[str, frozenset[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, variants in groups.items():
        for v in variants:
            out[normalize_relation(v)] = canon
    return out


_NORM_TO_CANON: dict[str, str] = _build_norm_to_canon_from_groups(RELATION_SYNONYM_GROUPS)


def load_relation_synonym_groups() -> dict[str, frozenset[str]]:
    """Merge built-in groups with optional RELATION_SYNONYMS_PATH JSON."""
    groups = {k: set(v) for k, v in RELATION_SYNONYM_GROUPS.items()}
    path = os.environ.get("RELATION_SYNONYMS_PATH", "").strip()
    if not path:
        return {k: frozenset(v) for k, v in groups.items()}
    p = Path(path)
    if not p.is_file():
        print(f"[pipeline] RELATION_SYNONYMS_PATH not found: {path}", flush=True)
        return {k: frozenset(v) for k, v in groups.items()}
    extra = json.loads(p.read_text(encoding="utf-8"))
    for canon, variants in extra.items():
        groups.setdefault(canon, set())
        for v in variants:
            groups[canon].add(normalize_relation(str(v)))
    return {k: frozenset(v) for k, v in groups.items()}


def relation_canonical_key(rel: str, norm_to_canon: dict[str, str] | None = None) -> str:
    n = normalize_relation(rel)
    if norm_to_canon is None:
        norm_to_canon = _NORM_TO_CANON
    return norm_to_canon.get(n, n)


def _relation_mode(relations: list[str]) -> str:
    if not relations:
        return ""
    counts = Counter(relations)
    best = counts.most_common()
    top_freq = best[0][1]
    candidates = [r for r, c in best if c == top_freq]
    return min(candidates, key=lambda x: (len(x), x))


def _strip_line_noise(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[-*•]\s+", "", s)
    s = re.sub(r"^\d+[.)]\s+", "", s)
    s = re.sub(r"^```\w*\s*", "", s)
    s = re.sub(r"```\s*$", "", s)
    return s.strip()


def _unwrap_relation_brackets(relation: str) -> str:
    s = relation.strip()
    m = re.fullmatch(r"<\s*(.+?)\s*>", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


def _extract_last_evidence_tag(line: str) -> tuple[str, str | None]:
    tags = list(re.finditer(EVIDENCE_TAG, line, re.I))
    if not tags:
        return line, None
    last = tags[-1]
    kind = "extracted" if last.group(1).lower() == "extracted" else "inferred"
    body = (line[: last.start()] + line[last.end() :]).strip()
    return body, kind


def _edge_from_regex_match(
    m_paren: re.Match[str] | None,
    m_plain: re.Match[str] | None,
    *,
    kind: str,
) -> ParsedEdge | None:
    m = m_paren or m_plain
    if not m:
        return None
    if m_paren:
        src_name, src_def, relation, dst_name, dst_def = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
            m.group(5),
        )
    else:
        src_name, relation, dst_name = m.group(1), m.group(2), m.group(3)
        src_def, dst_def = "", ""
    relation = _unwrap_relation_brackets(relation.strip())
    src_name = src_name.strip()
    dst_name = dst_name.strip()
    src_def = (src_def or "").strip()
    dst_def = (dst_def or "").strip()
    if not relation or not src_name or not dst_name or "*" in relation:
        return None
    return ParsedEdge(
        src_name=src_name,
        src_def=src_def,
        relation=relation,
        dst_name=dst_name,
        dst_def=dst_def,
        kind=kind,
    )


def _try_strict_line(line: str, *, kind: str | None = None) -> ParsedEdge | None:
    body, inferred_kind = _extract_last_evidence_tag(line)
    use_kind = kind or inferred_kind
    if use_kind is None:
        return None
    tag_suffix = "Extracted" if use_kind == "extracted" else "Inferred"
    candidate = f"{body} <{tag_suffix}>"
    m_paren = LINE_RE.match(candidate)
    m_plain = None if m_paren else LINE_RE_PLAIN.match(candidate)
    return _edge_from_regex_match(m_paren, m_plain, kind=use_kind)


def _find_entity_spans(body: str) -> list[tuple[int, int, str, str]]:
    """Non-overlapping *entity* spans: (start, end, name, category)."""
    spans: list[tuple[int, int, str, str]] = []
    for m in re.finditer(r"\*([^*]+?)\s*\(([^)]*)\)\*", body):
        spans.append((m.start(), m.end(), m.group(1).strip(), m.group(2).strip()))
    if len(spans) >= 2:
        return spans
    spans = []
    for m in re.finditer(r"\*([^*]+?)\*", body):
        spans.append((m.start(), m.end(), m.group(1).strip(), ""))
    return spans


def _try_loose_entities(body: str, *, kind: str) -> ParsedEdge | None:
    spans = _find_entity_spans(body)
    if len(spans) != 2:
        return None
    (_, e1_end, src_name, src_def), (e2_start, _, dst_name, dst_def) = spans[0], spans[1]
    relation = _unwrap_relation_brackets(body[e1_end:e2_start].strip())
    if not relation or not src_name or not dst_name or "*" in relation:
        return None
    return ParsedEdge(
        src_name=src_name,
        src_def=src_def,
        relation=relation,
        dst_name=dst_name,
        dst_def=dst_def,
        kind=kind,
    )


def _parse_single_relation_line(line: str) -> tuple[ParsedEdge | None, bool]:
    """Returns (edge, used_fallback). Fallback = any path beyond raw strict match."""
    edge = _try_strict_line(line)
    if edge:
        return edge, False

    norm = _strip_line_noise(line)
    if norm != line:
        edge = _try_strict_line(norm)
        if edge:
            return edge, True

    body, kind = _extract_last_evidence_tag(norm)
    if kind is None and "*" in norm:
        # Two entities but no tag: conservative default.
        kind = "inferred"
    if kind:
        for cand in (norm, body):
            if cand == line:
                continue
            edge = _try_strict_line(cand, kind=kind)
            if edge:
                return edge, True
        edge = _try_strict_line(body, kind=kind)
        if edge:
            return edge, True
        edge = _try_loose_entities(body, kind=kind)
        if edge:
            return edge, True
    return None, False


def parse_relation_lines(text: str, *, max_failure_samples: int = 5) -> tuple[list[ParsedEdge], ParseStats]:
    stats = ParseStats()
    edges: list[ParsedEdge] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            stats.skipped_empty += 1
            continue
        edge, used_fallback = _parse_single_relation_line(line)
        if edge:
            edges.append(edge)
            stats.ok += 1
            if used_fallback:
                stats.fallback_ok += 1
            continue
        if re.search(EVIDENCE_TAG, line, re.I) and "*" in line:
            stats.bad_format += 1
        else:
            stats.missing_tag += 1
        if len(stats.sample_failures) < max_failure_samples:
            stats.sample_failures.append(line[:200])
    return edges, stats


def _ensure_node(
    nodes_out: list[dict[str, Any]],
    key_to_id: dict[str, str],
    name: str,
    category: str,
) -> str | None:
    k = _node_key(name)
    if k in key_to_id:
        return key_to_id[k]
    nid = f"n{len(nodes_out):04d}"
    key_to_id[k] = nid
    nodes_out.append({"id": nid, "name": name.strip(), "category": category.strip()})
    return nid


EdgeGroup = dict[str, Any]  # relations: list[str], kinds: list[str]


def merge_parsed_edges_with_frequency(
    nodes_out: list[dict[str, Any]],
    edge_groups: dict[tuple[str, str, str], EdgeGroup],
    parsed: list[ParsedEdge],
    *,
    norm_to_canon: dict[str, str] | None = None,
) -> tuple[int, int]:
    """
    Mutates nodes_out and edge_groups. Returns (new_nodes_added, new_occurrences_added).
    Group key: (src_id, dst_id, relation_canonical_key).
    """
    key_to_id: dict[str, str] = {_node_key(n["name"]): n["id"] for n in nodes_out}
    nodes_before = len(nodes_out)
    occurrences = 0

    for pe in parsed:
        sid = _ensure_node(nodes_out, key_to_id, pe.src_name, pe.src_def)
        did = _ensure_node(nodes_out, key_to_id, pe.dst_name, pe.dst_def)
        if not sid or not did or sid == did:
            continue
        rkey = relation_canonical_key(pe.relation, norm_to_canon)
        gkey = (sid, did, rkey)
        if gkey not in edge_groups:
            edge_groups[gkey] = {"relations": [], "kinds": []}
        edge_groups[gkey]["relations"].append(pe.relation.strip())
        edge_groups[gkey]["kinds"].append(pe.kind)
        occurrences += 1

    return len(nodes_out) - nodes_before, occurrences


def finalize_edges_from_groups(
    edge_groups: dict[tuple[str, str, str], EdgeGroup],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for (sid, did, _rkey), g in sorted(edge_groups.items()):
        relations: list[str] = g["relations"]
        kinds: list[str] = g["kinds"]
        kind = "extracted" if any(k == "extracted" for k in kinds) else "inferred"
        row: dict[str, Any] = {
            "src": sid,
            "dst": did,
            "relation": _relation_mode(relations),
            "kind": kind,
        }
        freq = len(relations)
        if freq > 1:
            row["frequency"] = freq
        edges.append(row)
    return edges


def load_dotenv_simple() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


@dataclass
class Chunk:
    """In-pipeline chunk record. `chunk_id` is assigned later as c0000…"""
    chunk_id: str
    text: str
    source: str = ""           # readable label, e.g. "Tirofiban_info::pharmacodynamics"
    source_meta: dict = field(default_factory=dict)  # structured metadata


_HARD_BOUNDARIES = [
    re.compile(r"\n\s*\n+"),                 # 1) blank line
    re.compile(r"\n\s*[\*\-]\s+"),           # 2) bullet item ("\n    * item")
    re.compile(r"\n\s*\d+[.)]\s+"),          # 3) numbered item ("1. ", "2) ")
    re.compile(r"(?<=[.!?])\s+"),            # 4) sentence terminator
]


def _hard_wrap(seg: str, limit: int) -> list[str]:
    """Last-resort: split a too-long atom on word boundaries near `limit`."""
    if len(seg) <= limit:
        return [seg]
    out: list[str] = []
    buf: list[str] = []
    cur = 0
    for w in re.split(r"(\s+)", seg):  # keep separators
        if cur + len(w) > limit and cur > 0:
            out.append("".join(buf).strip())
            buf = []
            cur = 0
        buf.append(w)
        cur += len(w)
    if buf:
        out.append("".join(buf).strip())
    return [p for p in out if p]


def _split_sentences(text: str, hard_limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text into atomic segments by trying boundary regexes from coarse to fine.
    Any atom still longer than `hard_limit` is hard-wrapped on word boundaries.
    """
    segments: list[str] = [text.strip()]
    for rx in _HARD_BOUNDARIES:
        new_segments: list[str] = []
        for seg in segments:
            if len(seg) <= hard_limit:
                new_segments.append(seg)
                continue
            pieces = [p.strip() for p in rx.split(seg) if p and p.strip()]
            new_segments.extend(pieces if pieces else [seg])
        segments = new_segments

    out: list[str] = []
    for seg in segments:
        if len(seg) <= hard_limit:
            if seg:
                out.append(seg)
        else:
            out.extend(_hard_wrap(seg, hard_limit))
    return [s for s in out if s]


def _pack_segments(segments: Iterable[str]) -> list[str]:
    """
    Pack sentences into chunks <= MAX_CHUNK_CHARS.
    After packing, any trailing chunk shorter than MIN_CHUNK_CHARS is merged into the
    previous chunk so we don't keep almost-empty fragments like "Target Area Hair Count.".
    """
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            out.append("\n".join(buf).strip())
            buf = []
            buf_len = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if buf_len + len(seg) + 1 > MAX_CHUNK_CHARS and buf_len >= MIN_CHUNK_CHARS:
            flush()
        buf.append(seg)
        buf_len += len(seg) + 1
    flush()

    merged: list[str] = []
    for piece in out:
        if merged and len(piece) < MIN_CHUNK_CHARS:
            merged[-1] = (merged[-1] + "\n" + piece).strip()
        else:
            merged.append(piece)
    return merged


def _split_block(text: str) -> list[str]:
    """Split a block into chunk-sized texts (sentence-aware)."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text.strip()]
    return _pack_segments(_split_sentences(text))


def _texts_from_disease(text: str) -> list[tuple[str, dict]]:
    """Split disease_info block; extract disease name and sub-section labels."""
    # Extract disease name from first line like "* Information for 'large artery stroke':"
    disease_name = "unknown_disease"
    m = re.search(r"Information for ['\"]([^'\"]+?)['\"]", text)
    if m:
        disease_name = m.group(1).strip()

    # Split on "* Mayo SectionName:" or "* SectionName:" boundaries
    # Keep the section header with its content
    section_re = re.compile(r"\n(?=\*\s+(?:Mayo\s+)?[A-Z][A-Za-z\s]+:)")
    raw_blocks = section_re.split(text.strip())

    # Fallback: if no section splits found, try generic "* " bullet boundaries
    if len(raw_blocks) <= 1:
        raw_blocks = [b.strip() for b in re.split(r"\n(?=\*\s)", text.strip()) if b.strip()]

    out: list[tuple[str, dict]] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        # Identify section label from leading "* Mayo Symptoms:" or "* Symptoms:"
        sec_m = re.match(r"\*\s+(?:Mayo\s+)?([A-Z][A-Za-z\s]+?)\s*:", block)
        section = sec_m.group(1).strip().lower().replace(" ", "_") if sec_m else "general"
        meta = {"type": "disease", "entity": disease_name, "section": section}
        for chunk_text in _split_block(block):
            out.append((chunk_text, meta))
    return out


def _texts_from_drug(text: str) -> list[tuple[str, dict]]:
    """Split drug_info block per drug; extract drug name and sub-field labels."""
    # Split on each "* Name: DrugName" boundary
    drug_blocks = re.split(r"\n(?=\*\s*Name\s*:)", text.strip())
    out: list[tuple[str, dict]] = []
    for drug_block in drug_blocks:
        drug_block = drug_block.strip()
        if not drug_block:
            continue
        # Extract drug name
        name_m = re.match(r"\*\s*Name\s*:\s*(.+)", drug_block)
        drug_name = name_m.group(1).strip() if name_m else "unknown_drug"

        # Split on sub-field boundaries: "* FieldName:" (e.g. * Indication:, * Mechanism of Action:)
        field_re = re.compile(r"\n(?=\*\s+[A-Z][A-Za-z\s]+:)")
        sub_blocks = field_re.split(drug_block)

        for sub in sub_blocks:
            sub = sub.strip()
            if not sub:
                continue
            # Identify field label
            fld_m = re.match(r"\*\s+([A-Z][A-Za-z\s]+?)\s*:", sub)
            if fld_m:
                section = fld_m.group(1).strip().lower().replace(" ", "_")
            elif sub.startswith("* Name:"):
                section = "name"
            else:
                section = "general"
            meta = {"type": "drug", "entity": drug_name, "section": section}
            for chunk_text in _split_block(sub):
                out.append((chunk_text, meta))
    return out


def _texts_from_papers(text: str) -> list[tuple[str, dict]]:
    """Split relevant_papers block; extract PMID and title per paper."""
    parts = re.split(r"(?=\*\s*PMID\s+\d+)", text)
    out: list[tuple[str, dict]] = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 30:
            continue
        # Extract PMID
        pmid_m = re.match(r"\*\s*PMID\s+(\d+)", part)
        pmid = f"PMID {pmid_m.group(1)}" if pmid_m else "unknown_paper"
        # Extract title
        title_m = re.search(r"Title\s*:\s*(.+?)(?:\n|$)", part)
        title = title_m.group(1).strip() if title_m else ""
        meta = {"type": "paper", "entity": pmid}
        if title:
            meta["title"] = title
        for chunk_text in _split_block(part):
            out.append((chunk_text, meta))
    return out


def _texts_from_trials(text: str) -> list[tuple[str, dict]]:
    """Split similar_trials block; extract NCT ID and title per trial."""
    text = re.sub(r"^---\s*Similar\s+Phase\d+\s+Trials\s*---\s*", "", text.strip(), flags=re.I)
    trials = re.split(r"\n(?=\d+\.\s+Similar Trial\b)", text)
    out: list[tuple[str, dict]] = []
    for trial in trials:
        trial = trial.strip()
        if not trial or len(trial) < 40:
            continue
        # Extract NCT ID
        nct_m = re.search(r"NCT\s*ID\s*:\s*(NCT\d+)", trial)
        nct_id = nct_m.group(1) if nct_m else "unknown_trial"
        # Extract title
        title_m = re.search(r"\*\s*Title\s*:\s*(.+?)(?:\n|$)", trial)
        title = title_m.group(1).strip() if title_m else ""
        meta = {"type": "trial", "entity": nct_id}
        if title:
            meta["title"] = title
        for chunk_text in _split_block(trial):
            out.append((chunk_text, meta))
    return out


def _make_source_label(meta: dict) -> str:
    """Build a human-readable source label from structured metadata."""
    t = meta.get("type", "unknown")
    entity = meta.get("entity", "unknown")
    if t == "disease":
        slug = entity.lower().replace(" ", "_")
        section = meta.get("section", "general")
        return f"{slug}_info::{section}"
    elif t == "drug":
        section = meta.get("section", "general")
        return f"{entity}_info::{section}"
    elif t == "paper":
        slug_id = entity.replace(" ", "_")
        title = meta.get("title", "")
        title_slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60].strip("_") if title else ""
        return f"{slug_id}::{title_slug}" if title_slug else slug_id
    elif t == "trial":
        title = meta.get("title", "")
        title_slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60].strip("_") if title else ""
        return f"{entity}::{title_slug}" if title_slug else entity
    return entity


def build_chunks(profile: dict[str, Any]) -> list[Chunk]:
    # Tag each piece with (source_type, text, meta) for fine-grained provenance.
    pieces: list[tuple[str, str, dict]] = []
    if profile.get("disease_info"):
        for text, meta in _texts_from_disease(profile["disease_info"]):
            pieces.append(("disease", text, meta))
    if profile.get("drug_info"):
        for text, meta in _texts_from_drug(profile["drug_info"]):
            pieces.append(("drug", text, meta))
    if profile.get("relevant_papers"):
        for text, meta in _texts_from_papers(profile["relevant_papers"]):
            pieces.append(("paper", text, meta))
    if profile.get("similar_trials"):
        for text, meta in _texts_from_trials(profile["similar_trials"]):
            pieces.append(("trial", text, meta))

    # Dedup
    seen: set[str] = set()
    deduped: list[tuple[str, str, dict]] = []
    for src, t, meta in pieces:
        k = t.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append((src, k, meta))

    def _meta_key(m: dict) -> str:
        return f"{m.get('type', '')}|{m.get('entity', '')}|{m.get('section', '')}|{m.get('title', '')}"

    # Merge any chunk < MIN_CHUNK_CHARS into the previous chunk OF THE SAME SOURCE,
    # or into the next same-source chunk if it's the first short one.
    merged: list[tuple[str, str, dict]] = []
    for src, t, meta in deduped:
        if (
            len(t) < MIN_CHUNK_CHARS
            and merged
            and _meta_key(merged[-1][2]) == _meta_key(meta)
            and len(merged[-1][1]) + len(t) + 1 <= MAX_CHUNK_CHARS * 2
        ):
            merged[-1] = (src, merged[-1][1] + "\n" + t, meta)
        else:
            merged.append((src, t, meta))
    # Second pass: if a chunk is still short and the NEXT one is same source, fold forward.
    final: list[tuple[str, str, dict]] = []
    i = 0
    while i < len(merged):
        src, t, meta = merged[i]
        if (
            len(t) < MIN_CHUNK_CHARS
            and i + 1 < len(merged)
            and _meta_key(merged[i + 1][2]) == _meta_key(meta)
        ):
            merged[i + 1] = (src, t + "\n" + merged[i + 1][1], meta)
        else:
            final.append((src, t, meta))
        i += 1

    return [
        Chunk(
            chunk_id=f"c{idx:04d}",
            text=t,
            source=_make_source_label(meta),
            source_meta=meta,
        )
        for idx, (src, t, meta) in enumerate(final)
    ]


GRAPH_BUILD_SYS_LINE = """Build a knowledge graph for eligibility-criteria writing. Output parseable relation lines only.

== OUTPUT FORMAT (strict) ==
Output ONLY parseable lines. Each line MUST be exactly:
  *Entity1 (type label)* <relation> *Entity2 (type label)* <Extracted>
or end with <Inferred> instead of <Extracted>.

- Entity = text inside first/second asterisk pair; the parens carry a short TYPE LABEL
  that CLASSIFIES the entity, not a synonym or a narrower form of the name.
- The type label is OPTIONAL on each side. If you cannot name a precise type for a
  side, OMIT the parens on that side. Never leave empty parens.
- Direction: read left-to-right as TRUE: "Entity1 <relation> Entity2". Entity1 is src, Entity2 is dst.
- No JSON, no markdown fences, no bullet lists, no explanatory prose.

== NODES (what to capture) ==
Reusable concepts for writing eligibility: disease/phenotype, drug/intervention, outcome/endpoint, biomarker/lab/scale, contraindication, washout, screening rule, safety threshold, comorbidity; population cues (pregnancy, age band) when tied to a clinical fact.
The NAME carries the specific instance (a drug name, a condition, a threshold, a named rule). The TYPE LABEL in parens carries a fine-grained semantic role (e.g. "antihypertensive", "autoimmune disease", "inclusion criterion", "exclusion criterion", "biomarker", "washout period", "lab threshold", "comorbidity", "adverse event").
From similar trials or papers, lift the rule or finding — not the citation.

== NODES (avoid) ==
- Section headings or structural labels as node names. When a chunk heading points at a clinical rule, lift the specific fact (the drug, the condition, the threshold, the named criterion) — not the heading itself.
- Generic event/process words ("treatment", "intervention", "monitoring", "follow-up", "assessment") as standalone nodes unless tied to a specific procedure with a name and a measure.
- Parens that repeat the name, narrow it, or just rephrase it. The type label must classify the entity, not duplicate it.
- Empty parens. If you cannot name a precise type for a side, omit the parens on that side.

== NODES (never) ==
NCT/PMID/registry IDs, trial titles as placeholders, "Similar Trial N", study-record labels. ID-only mention → omit or restate the clinical concept.

== EVIDENCE TAGS (end of every line) ==
- <Extracted>: obvious literal support in THIS BATCH's chunks (same/neighboring sentence, direct statement).
- <Inferred>: entity names appear far apart in the batch but are reasonably linkable from the batch text only.
  Do NOT invent entities absent from the batch.
- Never tag <Inferred> on a relation that contradicts established clinical knowledge or sound medical judgment—omit the line instead.

== RELATION & DIRECTION ==
Relation is a plain English phrase between the two entities (no angle brackets around the relation).
NEVER self-loops. Say each triple aloud; fix reversed src/dst.
Typical: drug/treatment on LEFT, disease/outcome on RIGHT for treats/indicated-for patterns.
"""


def _anthropic_call_text(client: Any, model: str, system: str, user: str) -> str:
    max_out = int(os.environ.get("GRAPH_MAX_OUTPUT_TOKENS", "16384"))
    thinking_budget = int(os.environ.get("GRAPH_THINKING_BUDGET", "0"))
    base_kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_out,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    def _call(with_thinking: bool):
        if with_thinking and thinking_budget > 0:
            return client.messages.create(
                **base_kwargs,
                temperature=1.0,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
            )
        return client.messages.create(**base_kwargs, temperature=0.2)

    msg = None
    last_err: Exception | None = None
    attempts: list[tuple[bool, str]] = []
    if thinking_budget > 0:
        attempts.append((True, "thinking"))
    attempts.extend([(False, "plain"), (False, "plain-retry")])

    for idx, (with_thinking, label) in enumerate(attempts):
        try:
            msg = _call(with_thinking=with_thinking)
            break
        except Exception as e:
            last_err = e
            if idx < len(attempts) - 1:
                print(
                    f"[pipeline] LLM call failed ({label}, {type(e).__name__}: {e}); retrying...",
                    flush=True,
                )
                continue
            raise last_err from e

    def _text_from_msg(m: Any) -> str:
        return "\n".join(
            getattr(block, "text", "")
            for block in m.content
            if getattr(block, "type", None) == "text"
        ).strip()

    text = _text_from_msg(msg)
    if not text:
        print("[pipeline] empty text blocks; retrying once...", flush=True)
        msg = _call(with_thinking=False)
        text = _text_from_msg(msg)
    if not text:
        types = [getattr(b, "type", "?") for b in msg.content]
        stop = getattr(msg, "stop_reason", "?")
        raise RuntimeError(f"LLM returned no text (stop_reason={stop}, block_types={types})")
    return text


def extract_batch_round(
    client: Any,
    model: str,
    batch_chunks: list[Chunk],
    round_idx: int,
    total_rounds: int,
) -> str:
    chunk_txt = "\n\n".join(f"### {c.chunk_id}\n{c.text}" for c in batch_chunks)
    user = (
        f"ROUND {round_idx + 1} / {total_rounds}\n\n"
        "NEW_CHUNKS (sole evidence scope for this round — no prior graph):\n---\n"
        f"{chunk_txt}\n---\n\n"
        "Output parseable lines only. REQUIRED format per line:\n"
        "  *Entity1 (type label)* relation phrase *Entity2 (type label)* <Extracted>\n"
        "or the same with <Inferred> at the end. Relation is plain text (not wrapped in <>).\n"
        "Parens are optional per side: omit when no precise type fits. Clinical concepts only — not NCT/PMID or trial labels. (no prose, no JSON)"
    )
    text = _anthropic_call_text(client, model, GRAPH_BUILD_SYS_LINE, user)
    if not text:
        raise RuntimeError(f"Batch round {round_idx + 1}: empty model response.")
    return text


def neighbor_phrases(
    node_idx: int,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_edges: int = 14,
) -> list[str]:
    phrases: list[str] = []
    nid = nodes[node_idx]["id"]
    name_of = {n["id"]: n["name"] for n in nodes}
    for e in edges:
        if e["src"] == nid:
            phrases.append(f'{e["relation"]} -> {name_of.get(e["dst"], e["dst"])}')
        elif e["dst"] == nid:
            phrases.append(f'{name_of.get(e["src"], e["src"])} -> {e["relation"]}')
        else:
            continue
        if len(phrases) >= max_edges:
            break
    return phrases


def run_pipeline(
    profile_path: str,
    out_path: str,
    *,
    batch_size_cli: int | None = None,
) -> None:
    load_dotenv_simple()

    if Anthropic is None:
        raise SystemExit("Install anthropic: pip install -r requirements-graph.txt")

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY must be set (see API.md).")

    graph_model = os.environ.get("GRAPH_MODEL", "MiniMax-M2.7")

    embed_backend = os.environ.get("EMBEDDING_BACKEND", "minimax_rest").strip().lower()
    if embed_backend == "sentence_transformers":
        embed_model_label = os.environ.get(
            "ST_EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    else:
        embed_model_label = os.environ.get("MINIMAX_EMBEDDING_MODEL", "emb-o01")

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    chunks = build_chunks(profile)
    if not chunks:
        raise SystemExit("No chunks produced from profile.")

    timeout_sec = float(os.environ.get("ANTHROPIC_TIMEOUT_SEC", "600"))
    client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    batch_sz = batch_size_cli
    if batch_sz is None:
        batch_sz = int(os.environ.get("GRAPH_CHUNK_BATCH_SIZE", "10"))
    batch_sz = max(1, batch_sz)
    nodes_out: list[dict[str, Any]] = []
    edge_groups: dict[tuple[str, str, str], EdgeGroup] = {}
    synonym_groups = load_relation_synonym_groups()
    norm_to_canon = _build_norm_to_canon_from_groups(synonym_groups)
    min_lines = int(os.environ.get("GRAPH_MIN_LINES_PER_ROUND", "0"))

    structured_tool_label = "plain_text_line_format_with_evidence_tags"
    n_rounds = max(1, (len(chunks) + batch_sz - 1) // batch_sz)
    print(
        f"[pipeline] chunks={len(chunks)} (batch_size={batch_sz}) — {n_rounds} line-format LLM round(s)...",
        flush=True,
    )

    for ri, start in enumerate(range(0, len(chunks), batch_sz)):
        batch = chunks[start : start + batch_sz]
        print(
            f"[pipeline] round {ri + 1}/{n_rounds} — LLM on {len(batch)} chunks (timeout={timeout_sec}s)...",
            flush=True,
        )
        raw = extract_batch_round(client, graph_model, batch, ri, n_rounds)
        parsed, pstats = parse_relation_lines(raw)
        fb = f", fallback={pstats.fallback_ok}" if pstats.fallback_ok else ""
        if pstats.ok == 0 and min_lines > 0:
            print(
                f"[pipeline]   WARNING: 0 parseable lines (min={min_lines}); "
                f"bad={pstats.bad_format} missing_tag={pstats.missing_tag}{fb}",
                flush=True,
            )
        elif pstats.ok == 0:
            print(
                f"[pipeline]   parse: 0 ok, bad={pstats.bad_format}, missing_tag={pstats.missing_tag}{fb}",
                flush=True,
            )
        else:
            print(
                f"[pipeline]   parse: {pstats.ok} ok, bad={pstats.bad_format}, "
                f"missing_tag={pstats.missing_tag}{fb}",
                flush=True,
            )
        nn, noc = merge_parsed_edges_with_frequency(
            nodes_out, edge_groups, parsed, norm_to_canon=norm_to_canon
        )
        print(
            f"[pipeline]   merged +{nn} nodes, +{noc} occurrences "
            f"(totals {len(nodes_out)}n {sum(len(g['relations']) for g in edge_groups.values())} occ)",
            flush=True,
        )

    edges_out = finalize_edges_from_groups(edge_groups)

    if not nodes_out:
        raise SystemExit("LLM produced zero nodes across all rounds.")

    chunk_store = {c.chunk_id: {"text": c.text, "source": c.source, "source_meta": c.source_meta} for c in chunks}
    skip_embed = os.environ.get("GRAPH_SKIP_EMBED", "").strip().lower() in ("1", "true", "yes")

    if skip_embed:
        print("[pipeline] GRAPH_SKIP_EMBED set — skipping embeddings; chunk_ref omitted.", flush=True)
        artifact_steps = [
            "A_short_chunks_dedup_sequential_ids",
            "B_llm_line_format_tags_no_carry_forward_chunk_batches",
            "C_merge_synonym_relation_frequency_stable_node_ids_drop_selfloops",
            "D_skipped_embeddings",
        ]
        artifact = {
            "schema": "trial_knowledge_graph_v4",
            "source_profile": os.path.basename(profile_path),
            "pipeline": {"chunk_batch_size": batch_sz, "steps": artifact_steps},
            "models": {
                "llm": {
                    "sdk": "anthropic",
                    "base_url": base_url,
                    "model": graph_model,
                    "structured_output": structured_tool_label,
                },
            },
            "notes": {
                "nodes": "category = parenthetical short phrase from line format.",
                "edges": "kind from <Extracted>/<Inferred>; frequency = cross-batch/synonym count (omitted when 1).",
            },
            "chunks": chunk_store,
            "nodes": nodes_out,
            "edges": edges_out,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        print(
            f"OK: {len(nodes_out)} nodes, {len(edges_out)} edges, {len(chunks)} chunks -> {out_path}",
            flush=True,
        )
        return

    texts = [c.text for c in chunks]
    print(f"[pipeline] embedding {len(texts)} chunks...", flush=True)
    chunk_mat = embed_batch(texts, "db")
    if chunk_mat.shape[0] != len(texts):
        raise RuntimeError("Embedding matrix row count mismatch with chunks.")

    query_txts: list[str] = []
    for i, node in enumerate(nodes_out):
        nei = neighbor_phrases(i, nodes_out, edges_out)
        parts = [node["name"]]
        if node.get("category"):
            parts.append(node["category"])
        if nei:
            parts.append("; ".join(nei))
        query_txts.append(" | ".join(parts))
    print(f"[pipeline] embedding {len(query_txts)} node queries (single batch)...", flush=True)
    q_mat = embed_batch(query_txts, "query")
    if q_mat.shape[0] != len(nodes_out):
        raise RuntimeError("Query embedding row count mismatch with nodes.")
    for i, node in enumerate(nodes_out):
        top_i = cosine_top1(q_mat[i], chunk_mat)
        node["chunk_ref"] = chunks[top_i].chunk_id

    chunk_store = {c.chunk_id: {"text": c.text, "source": c.source, "source_meta": c.source_meta} for c in chunks}

    artifact_steps = [
        "A_short_chunks_dedup_sequential_ids",
        "B_llm_line_format_tags_no_carry_forward_chunk_batches",
        "C_merge_synonym_relation_frequency_stable_node_ids_drop_selfloops",
        "D_embed_chunks_db_embed_node_queries_query_cosine_top1_chunk_ref",
    ]

    artifact = {
        "schema": "trial_knowledge_graph_v4",
        "source_profile": os.path.basename(profile_path),
        "pipeline": {
            "chunk_batch_size": batch_sz,
            "steps": artifact_steps,
            "similarity": {
                "metric": "cosine_similarity",
                "vector_normalize": "l2_rowwise_before_dot_product",
                "selection": "argmax_strict_top1_no_threshold",
            },
        },
        "models": {
            "llm": {
                "sdk": "anthropic",
                "base_url": base_url,
                "model": graph_model,
                "structured_output": structured_tool_label,
            },
            "embedding": {
                "backend": embed_backend,
                "model": embed_model_label,
                "chunk_embed_type": "db",
                "query_embed_type": "query",
            },
        },
        "notes": {
            "nodes": (
                "Concept-oriented for eligibility-criteria writing. `category` holds the parenthetical "
                "short phrase from line-format extraction. No case identifiers (NCT/PMID)."
            ),
            "edges": (
                "Free-form relations with kind from line suffix <Extracted>/<Inferred>. "
                "frequency counts occurrences across batches (synonymous relations merged). "
                "Omitted when 1 (default). "
                "kind aggregates to extracted if any occurrence was extracted."
            ),
            "chunk_ref": (
                "Top-1 cosine chunk per node using query = name | category | 1-hop edge phrases."
            ),
        },
        "chunks": chunk_store,
        "nodes": nodes_out,
        "edges": edges_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(nodes_out)} nodes, {len(edges_out)} edges, {len(chunks)} chunks -> {out_path}", flush=True)


def run_self_tests() -> None:
    """Parse + merge unit checks (no LLM)."""
    sample = (
        "*NSCLC (Non Small Cell Lung Cancer)* is treated by "
        "*Gefitinib (A tumor targeted drug)* <Extracted>\n"
        "*NSCLC (lung cancer)* is treated by *Gefitinib (EGFR drug)* <extracted>\n"
        "*Hypertension (comorbidity)* linked to *Screening burden (process)* <Inferred>\n"
        "- *DrugA (drug)* <treats> *DisB (disease)* <Extracted>\n"
        "*Foo (a)* relates *Bar (b)* <Inferred> trailing note\n"
        "*Gap (a)* bridges *GapB (b)*\n"
        "bad line without tag\n"
    )
    parsed, stats = parse_relation_lines(sample)
    assert stats.ok == 6, stats
    assert stats.fallback_ok >= 2, stats
    assert parsed[0].kind == "extracted"
    assert parsed[2].kind == "inferred"
    assert parsed[3].relation == "treats"
    assert parsed[4].kind == "inferred"
    assert parsed[5].relation == "bridges"
    assert parsed[5].kind == "inferred"

    nodes: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], EdgeGroup] = {}
    norm = _build_norm_to_canon_from_groups(load_relation_synonym_groups())
    merge_parsed_edges_with_frequency(
        nodes,
        groups,
        [ParsedEdge("DrugA", "drug", "treated by", "DisB", "disease", "extracted")],
        norm_to_canon=norm,
    )
    edges_one = finalize_edges_from_groups(groups)
    assert len(edges_one) == 1 and "frequency" not in edges_one[0]
    merge_parsed_edges_with_frequency(
        nodes,
        groups,
        [ParsedEdge("DrugA", "drug", "therapy with", "DisB", "disease", "inferred")],
        norm_to_canon=norm,
    )
    edges = finalize_edges_from_groups(groups)
    assert len(edges) == 1, edges
    assert edges[0]["frequency"] == 2
    assert edges[0]["kind"] == "extracted"

    groups2: dict[tuple[str, str, str], EdgeGroup] = {}
    nodes2: list[dict[str, Any]] = []
    merge_parsed_edges_with_frequency(
        nodes2,
        groups2,
        [ParsedEdge("X", "", "causes", "Y", "", "inferred")],
        norm_to_canon=norm,
    )
    merge_parsed_edges_with_frequency(
        nodes2,
        groups2,
        [ParsedEdge("X", "", "leads to", "Y", "", "extracted")],
        norm_to_canon=norm,
    )
    e2 = finalize_edges_from_groups(groups2)
    assert len(e2) == 1 and e2[0]["frequency"] == 2 and e2[0]["kind"] == "extracted"
    print("OK: run_self_tests passed", flush=True)


def write_chunks_preview(profile_path: str, out_path: str) -> None:
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    chunks = build_chunks(profile)
    preview = {
        "chunk_count": len(chunks),
        "chunks": [
            {"chunk_id": c.chunk_id, "source": c.source, "source_meta": c.source_meta, "text": c.text}
            for c in chunks
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
    print(f"Wrote chunk preview ({len(chunks)} chunks) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multidimensional graph from trial_profile JSON.")
    ap.add_argument("profile_json", nargs="?", help="Path to trial_profile *.json")
    ap.add_argument("-o", "--output", default="trial_graph.json", help="Output graph JSON path")
    ap.add_argument(
        "--chunks-preview",
        metavar="OUT_JSON",
        help="Chunk splitting only (no LLM / no embeddings).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Chunks per LLM round (default: env GRAPH_CHUNK_BATCH_SIZE or 10).",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run parse/merge unit checks and exit.",
    )

    args = ap.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.chunks_preview:
        if not args.profile_json:
            ap.error("profile_json is required with --chunks-preview")
        write_chunks_preview(args.profile_json, args.chunks_preview)
        return
    if not args.profile_json:
        ap.error("profile_json is required")
    run_pipeline(args.profile_json, args.output, batch_size_cli=args.batch_size)


if __name__ == "__main__":
    main()
