Wide 16:9 ACL / NeurIPS / ICML–style **method figure**: flat vector, white background, generous whitespace, colorblind-safe **teal** (indexing / store), **amber** (LLM), **muted purple** (Expert agent region), warm **gray** (passage strips); matte, no gloss, no 3D.

**Soft layout (guide only—vary geometry, spacing, and routing freely):** imagine a **construction stream** feeding a **central graph** as the visual anchor, with **per-node retrieval hooks** hanging under or beside that graph, and a **separate Expert-agent band** (lightly purple frame or tint) that branches from the graph for *per-subtask* reasoning. Reading order should feel like one pipeline: **corpus → index → iterative schema LLM (graph grows) → graph → top‑1 anchors → expert disclosure chain**. Arrows carry the story; avoid a rigid two-column table of equal boxes.

**What to draw, in logical order (match the paper pipeline; labels on canvas stay short):**

1. **Session intake:** header “Multi-domain RAG corpus (session)” plus small domain chips: *disease · trials · literature · drug*.

2. **A — CHUNK INDEX:** fine passage splits, stacked strip or deck motif flowing into a **vector / embedding store** (cylinder or index icon, teal).

3. **B — RELATION LLM (chunk batches only):** repeated amber block on **next chunk batch only** (no carry-forward graph) → **line-format** triples ending in **`<Extracted>` / `<Inferred>`** tags; optional loop motif for multiple rounds.

4. **C — SESSION GRAPH:** code parse + **single-line fallback** + **synonym relation merge**; edges show **kind** and **`frequency ×n` only when n > 1**; medium network as focal element.

5. **D — TOP‑1 passage anchor (per node):** for representative nodes only, show **name plus first‑hop incident-edge phrases** (relation / kind / neighbor cues) feeding retrieval into **one thin anchored excerpt strip** per node (pin or clip), not a wall of text.

6. **Per subtask — Expert agent · progressive disclosure** (purple-tinted zone):  
   **(1) COMPACT ENTRY** — horizontal **name roster** only (no edges, no raw passages).  
   **(2) SUBGRAPH EXTRACTION** — **task-specific seeds**, then **incident edges and one-hop neighbors** on the induced vertex set → a **smaller local subgraph** (magnifier or inset OK).  
   **(3a) CONTEXT — structure** — that local graph with **relation** labels and **extracted / inferred** styling consistent with the legend.  
   **(3b) CONTEXT — evidence** — **raw passages** linked to the active nodes in that subgraph.  
   Close with a compact **Expert agent reasoning** node or terminal consuming **(3a) then (3b)**.

**Figure copy rules:** English, **technical stack terms on the figure** (chunk index, schema LLM, extracted, inferred, TOP‑1 passage anchor, compact entry, subgraph extraction, progressive disclosure, Expert agent). Phrases of **a few words** per callout; **no long paragraphs** inside the art; **no slide-deck bullets**. **No variable names or formulas on canvas**—use short plain labels (e.g. “session graph”, “local subgraph”).