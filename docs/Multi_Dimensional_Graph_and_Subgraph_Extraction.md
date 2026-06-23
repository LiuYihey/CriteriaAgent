# Multi-Dimensional Graph Construction & Agentic Subgraph Extraction

本文档固定在 **3.2.1「多维图谱构建与动态子图抽取」** 语境下：承接多域 RAG 碎片，经由建图与子图抽取 **仅向 Expert agents 提供图上知识**（与 Overview 中虚线一致）；Task Planner / Criteria Writer / Reviewer 的接口详见 §3.6。此处只详述 **建图 + Expert agent 侧子图投喂**。

---

## 1. 问题与思路（创新精髓）

**痛点**：异构碎片直接拼进上下文易导致「上下文迷失」与跨段语义难以组织。**做法**：在中间层放一个 **会话级概念关系图**：节点以可读 **`name`** 标识，概念角色由 **关联边与局部拓扑** 体现（与 artifact 中「node 由边定义」一致）；边集上同时容纳 **有直接文本支撑的抽取** 与在各轮语料与会话已累计图结构支撑下所作的 **延申推断**，二者用 **`extracted` / `inferred` 二分标签**区分。**每个节点**再通过 **向量检索** 挂载 **Top‑1 短 chunk**，多节点共用同一 chunk 时用 **索引去重**。下游 **Expert agent（专家智能体）** 在 **渐进披露的子图 + 去重正文** 上精读，仍可继续推理。

**与「多维」**：图仍是 **二维拓扑 (V, E)**；**「多维」指证据与关系维度的交织**，具体包括：(i) **多域异构语料汇入**（疾病 / 试验 / 文献 / 药物等在同一图中连通）；(ii) **自由表述的多元关系 `relation`**（可不设封闭关系词表）；(iii) **边上 `extracted`/`inferred` 二分**（字面锚点 vs 语料内延申）。架构保持扁平：单一会话图，**关系与边上 kind 承载主要语义**。

**与「抽取 / 推断」**：图谱既要 **钉住字面可复述** 的连接，也允许 LLM 在 **当批 chunk** 内做 **跨段语义衔接**：行尾 **`<Extracted>`** 表示同批中有明显字面证据；**`<Inferred>`** 表示实体相隔较远但可合理关联，且 **不得输出违背 established clinical knowledge 的推断边**（应直接省略该行）。**跨批重复边** 由代码按 **同义 relation** 合并；**`frequency > 1`** 时才在边上记录出现次数。**Expert agent（专家智能体）** 在结合附录 raw 精读时，仍可对 **`inferred` 边** 自行把关。

---

## 2. Multi-Dimensional Graph Construction（建图流水线）

### 2.1 输入与前提

输入为 **单次试验会话下** 多域 RAG 汇总的文本材料（PrimeKG / ClinicalTrials.gov 相似试验 / 文献摘要 / DrugBank 等字段），形式上可为 JSON 或多个纯文本段落。

在此之前应已有 **Trial Configuration**（标题、治疗方案、终点等）用于上游检索；建图模块视这些材料 **与检索结果 jointly** 作为 schema 抽取的输入（具体拼接策略由工程决定，语义上是一段「可查证的语料合集」）。

### 2.2 Step A — 细粒度 Chunk 索引

**目标**：让每个节点稍后对齐到的 **raw 段落短而准**，且 **多节点可共用同一短文**，以 **chunk_id** 索引、不重复存储。

- 按来源与语义 **细粒度切分**：疾病按条目或短句群；药物按字段或 bullet；文献按题名/摘要小句群；相似试验按 NCT 子块（如 criteria / intervention / outcome）。
- 每条 chunk：`chunk_id`、`source`、`text`（建议对单块长度设上限，使「一段一意」）。
- 对 `text` 建立 **向量索引**（embedding 模型需中英文与医学术语兼备；最终以小规模 dev 上的 top‑1 命中率选型）。

### 2.3 Step B — 分批行格式 LLM（无 carry-forward）

**多轮**纯文本调用：**每轮仅**向模型投喂 **下一批短文块**（批次大小 \(N\) 可调，默认 10），**不**附带已累计图。模型每行输出一条三元组：

`*Entity1 (short phrase)* relation phrase *Entity2 (short phrase)* <Extracted|Inferred>`

- **节点**：星号内为 **`name`**，括号内简释写入 **`category`**（代码解析；同名归一后由代码分配稳定 **`id`**）。**不得**使用 NCT/PMID/试验元数据标识作为节点（prompt 约束，表达底层临床事实即可）。
- **方向**：从左到右读作真，左为 `src`、右为 `dst`。
- **`relation`**：两实体之间的 **纯英文短语**，**不用**尖括号包裹（避免与行尾 `<Extracted>`/`<Inferred>` 混淆）。
- **行尾标签**：**`<Extracted>`** = 当批 chunk 有明显字面证据；**`<Inferred>`** = 当批内实体相隔较远但可合理关联（不得臆造未出现的实体；**不得**违背临床医学常识——荒谬关联应省略整行）。
- **输出约束**：仅可解析行，无 JSON / 无 markdown 代码块 / 无解释段落。

### 2.4 Step C — 解析、同义合并与频次

代码 **正则解析** 各行（**严格匹配 + 单行 fallback**：去 `-`/编号前缀、剥 relation 外围 `<>`、取行内最后一个证据标签；无标签但可抽出两实体时默认 **`inferred`**；relation 中含额外 `*` 的三实体句丢弃）→ 写入会话图；**跨批 merge** 时：

- 边分组键：`(src_id, dst_id, relation_canonical_key)`，其中 relation 经 **表面归一 + 同义组**（如 treated by / therapy with）折叠；
- **`frequency`**：该组在全部批次中的出现次数（含同义表述）；**仅当 > 1 时写入边 JSON**（为 1 时省略，读作默认单次）；
- **`kind` 聚合**：组内任一 occurrence 为 `extracted` 则最终为 `extracted`，否则 `inferred`；
- **`relation` 展示**：取组内出现最多的原始表述。

得到 **会话级全局小图** 后进入 Step D。

### 2.5 Step D — 节点 → Chunk：检索 Query 设计与 Top‑1

对每个节点 \(v\) 构造 **检索 Query**（建议固定拼装顺序以便复现实验），使相似度最接近「与该节点在剧中角色一致」的短段落：

1. **`name_v`**
2. **一阶邻边摘要**：将与 \(v\) 相连的边写成短语（典型：`relation[kind] -- NeighborName`），即用 **拓扑上下文** 补足单靠名称可能模糊的语义（度大时截断 Top‑K 或按 `extracted` 优先等策略由工程约定）

将该 query（例如 `name_v | 邻边短语拼接`）embedding 后，在 Step A 向量库中取 **全局 Top‑1 chunk**，记录 **`chunk_ref(v) → chunk_id`**。

- **无阈值、无占位**：始终一条最相似 chunk。  
- **多节点指向同一 chunk_id**：仅 **引用**，文本只存一份；后面给 Expert agent 时 **按 chunk 去重**。  
节点侧 **不推荐**再引入 `canonical_label`；**`name` + `id`（工程）** 足够对外标识。

至此：**会话多维图谱 = (V, E，边上带 `extracted`|`inferred`) + 每节点 `{ id（代码生成）, name, chunk_ref }`**；**「多维」见 §1**，拓扑仍为二维平面上的图。

---

## 3. Agentic Subgraph Extraction（渐进披露 + Expert agent 选型 + 1-hop）

与设计目标对齐：**不传全文**；先看 **可调焦的地图**，再展开 **小范围拓扑 + 去重证据块**。

**何为 L0 / L1 / L2**：指 **Expert agent 侧** **渐进式披露** 的三次「打开程度」，全部由工具/网关按顺序拼进上下文——**只为 Expert agent**，不是别处用的协议版本号。

| 代号 | 名称 | Expert agent 收到的内容 |
|------|------|------------------|
| **L0** | **目录（Catalog）** | 全图节点 **`id`、`name` 列表**（紧凑表）。**没有边、没有正文 chunk**。用于子任务对齐后「点菜」要选哪些节点。 |
| **L1** | **子图拓扑** | 在 **Expert agent** 选型并经 **unique + 1-hop** 诱导后的 **`V'`、`E'`**：节点 **`id`、`name`** + **边**（含 `relation` 与 **`extracted`/`inferred`**；**`frequency` 仅 > 1 时出现**）。**仍不附正文 chunk**。 |
| **L2** | **去重正文附录** | 子图 **`V'`** 涵盖节点所挂载的 **`chunk_ref` → chunk 文本**：先 **按 `chunk_id` 去重**，整块作为附录；与前一步 **先 L1 后 L2** 投喂，以免未建立结构前就淹没在原文里。 |

「**经由 L0→L2**」即：**先有目录选型 → 再展开拓扑 → 再给去重之后的 raw**，三步依次消费完毕再形成 **Expert agent** 作答。

### 3.1 披露层 L0 — 图目录（Catalog）

向 **Expert agent（专家智能体）** 提供 **紧凑列表**：**服务端已分配好的 `id` + `name`**，**不包含**边与 raw。用途：与子任务对齐，决定要「点开」哪些节点。

### 3.2 Expert agent 选型与节点去重

**Expert agent（专家智能体）** 根据子任务输出 **`selected_ids`**。**服务端必须先 `unique(selected_ids)`**，再进入拓扑展开，杜绝模型重复递交同一节点。

### 3.3 仅 1-hop 诱导子图

记 **服务端对 Expert agent 输出的 `selected_ids` 做 `unique` 之后的节点集合** 为「**已选集**」。将图视作 **无向**（或沿每条边双向等价）时：**一阶子图顶点** = 已选集中的所有顶点 ∪ 与已选集中至少一顶相邻的所有顶点；再对顶点 id 做一次 **集合去重**。**子图的边** = 原图中 **两端都属于上述顶点集** 的那些边。因而在顶点层面 **每个节点 id 至多出现一次**。

若有 **完全相同的边**，可按需折叠或由下游决定是否保留。**不设**默认 RWR / 多跳；子任务重心在目录阶段的 **选型**，1-hop 只 **补齐一阶邻居**。

### 3.4 披露层 L1 — 子图结构

向 **Expert agent** 投喂 **`V'` 上节点摘要（`id`、`name`）与 `E'`**（每条边含 `relation` 与 **`extracted` / `inferred` 标签**），即 **显性结构**。仍 **不重复**附录 raw。

### 3.5 披露层 L2 — Raw Chunk 附录（按索引去重）

- 对每个 \(v\in V'\)，取 `chunk_ref(v)`得到 `chunk_id` 集合。
- **`unique(chunk_id)`** 后，按 id 拉出 `text`，作为 **单次附录**。  
顺序建议：**先 L1，后 L2**，避免 Expert agent 在未建立拓扑前先淹没问题。

若需压 token，可作 **消融策略**：raw **仅对已选顶点**展开全文，对 **仅由 1-hop 引入的邻居顶点**只保留拓扑、不附录 chunk（主干仍以 **本子图全体顶点** + **chunk 去重** 为准）。

### 3.6 与整体系统衔接（对齐 Overview）

图示数据流可作如下 **边界约定**（亦为本文档与方法图一致的前提）：

- **多域 RAG → Multi-Dimensional Graph Construction**：碎片化检索结果在本次会话中建图；**图谱及其 chunk 附录不作为全系统共用底座向下游泼水**。
- **Agentic Subgraph Extraction**（图中的紫色虚线）：**仅连通「建图产物」与各 Expert agent**。也就是说，**只有专家智能体（Expert agent）**经 L0→L2 **读取拓扑与去重 raw**；Task Planner、Criteria Writer、Criteria Reviewer **不直接接收整图或未抽取的全量图谱上下文**。
- **Task Planner**：以 **Trial Configuration**（标题、治疗方案、终点等）及监管指南为输入，拆解子任务并指派 **Expert agent**；**驱动的是子任务语义**，**Expert agent** 接单后再向图检索侧拉取 catalog/子图。
- **Expert agent（专家智能体）**：建图+RAG-derived 知识的 **唯一图上消费者**（本子方法的核心）；产出子任务级结论文本或结构化要点。
- **Least-to-Most / Criteria Writer**（图示）：汇入 **Expert agent 产出 + Trial Configuration + Writing Protocols**，整合为纳入/排除等标准草案。
- **Criteria Reviewer（LLM-as-Judge）**：**仅**评审 **Criteria Writer 输出的纳入/排除等待选文稿**；评分与文字反馈 **严格依据** 仓库内 [`Criteria reviewer.md`](Criteria%20reviewer.md) 的 **五维 rubric**（各维 1–10 分：**Signal Enrichment、Population Reach、Data Evaluability、Clinical Feasibility、Risk Mitigation**）并输出 **针对性 comments**。**不**另行检查图谱、`inferred`/raw 或其它非文稿对象。

**实现备注**：可采用 LangChain Tooling（outline / expand_1hop / attach_unique_chunks）；本文档不绑定具体框架 API。

---

## 4. 设计原则一览（去繁）

| 项 | 做法 |
|----|------|
| 多维 | **多域汇入 + 多元 `relation` + 边上 `extracted`/`inferred`** |
| 边 | **`extracted`**（字面/紧密对应）与 **`inferred`**（全料延申）；**不配 span、不配 support_chunk_id** |
| 节点 `id` | **代码分配**；schema **不写 `id`**；**同源节点去重交由 LLM（schema 一行一实体）** |
| 节点标识 | 对外：**`name` + `id`**（`id` 仅工程与选点） |
| Raw | **细 chunk + Query(name, 一阶邻边短语) → Top‑1** |
| Raw 存储 | **chunk_id 引用**；多处共享 |
| 子图 | **Expert agent unique 选型 + 1-hop + `V'` unique** |
| **Expert agent** 投喂 | **先图结构，再去重 chunk 文本** |

---

## 5. *科研绘图参考模板（ASCII）*

下列 ASCII（**全英文**）概括 **Construction (A→D)** 与 **per-subtask Expert agent · progressive disclosure ((1)→(2)→(3a)→(3b))**；版式上可作 **Fig. X** 的内容骨架，具体布局由排版/生图自适应。

```
  Multi-domain RAG corpus (session)
  domains: disease · trials · literature · drug
                    |
                    v
  A   CHUNK INDEX — fine passage splits, embedding store / vector index
                    |
                    v
  B   RELATION LLM — chunk batches only → line format + <Extracted>/<Inferred> tags
                    |
                    v
  C   SESSION GRAPH — code parse; synonym merge; edge frequency + kind
                    |
      per node: name + first-hop incident-edge phrases (relation, kind, neighbor) → retrieval
                    |
                    v
  D   TOP-1 passage anchor per node (single supporting excerpt)
                    |
      -------- per subtask · Expert agent · progressive disclosure --------
                    |
  (1) COMPACT ENTRY — node-name roster only (readable titles; no edges, no raw passages)
                    |
      task-specific seed selection
                    |
                    v
  (2) SUBGRAPH EXTRACTION — expand seeds: incident edges and one-hop neighbors
      on the induced vertex set → local subgraph for the Expert agent
                    |
                    v
  (3a) CONTEXT (structure) — local subgraph with relation labels; edge kind extracted / inferred
                    |
                    v
  (3b) CONTEXT (evidence) — raw passages linked to nodes in the local subgraph
                    |
                    v
      Expert agent reasoning
```

**图中可强调的「笔尖」用词（caption 级）：** 多域汇入 • **分批行格式建图（chunk batches only，行尾 Extracted/Inferred）** • **同义 relation 合并 + frequency** • **名称 + 一阶邻边短语 → Top‑1 段落锚定** • **名称入口 · 任务相关选点 · 诱导子图（选点 + 一阶闭包）** • **渐进式披露：先子图结构、后 raw 正文**。

---

## 6. 可写入论文的一段话（方法与贡献锚点）

> 我们将多域检索得到的碎片化语料先做 **细粒度可索引短文分块** 并建立向量索引；继之以 **分批行格式建图**：每一轮仅投喂 **下一批短文块**（默认每批 \(N\) 条），LLM 输出 `*E1 (phrase)* relation *E2 (phrase)* <Extracted|Inferred>` 行，代码 **严格解析 + 单行 fallback** 后 **按同义 relation 合并**，**`frequency > 1`** 时记录出现次数并聚合 **`kind`**。**每个节点**再经由 **`name` 与一阶邻边短语（含 relation/kind）拼接** 的检索查询绑定 **Top‑1 短文**。**Expert agent** 侧采用 **渐进式披露**：目录 → **诱导子图（含 kind；frequency 仅重复边）** → **去重正文**。**建图与子图投喂仅对接各 Expert agent**；下游经 **Expert 产出 + Trial Configuration + 撰写协议** 耦合。

---

*文档版本：与 `trial_graph.json`（`trial_knowledge_graph_v4`）一致——边 **`frequency` 仅 > 1 时写入**；行尾 **`<Extracted>`/`<Inferred>`** 解析为 **`kind`**（Inferred 须临床 plausible）；Step B 为 **无 carry-forward 行格式 LLM + 代码 merge/单行 fallback**；Step D 默认 **本地 sentence-transformers（如 BGE）** 或 MiniMax embedding（见 `.env.example`）。*
