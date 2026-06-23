## 3.1 试验配置与多域检索增强生成 (Trial Configuration & Multi-domain RAG)

为确保模型在生成标准时具备全面且实时的医学先验知识，本模块首先接收用户定义的**试验配置 (Trial Configuration)**，包括：

1. **临床试验标题 (Clinical Trial Title)**：明确研究主题；
2. **治疗方案 (Treatment Protocol)**：包含干预措施 (Intervention) 与参与者分组 (Participant Arm)；
3. **主要终点 (Primary Endpoint)**：包含结局指标 (Outcome Measure) 与时间框架 (Time Frame)。

以上述配置为Query，系统触发**多域检索增强生成 (Multi-domain RAG)**，从四个权威且异构的医学数据库中并行召回相关知识：

- **PrimeKG (精准医学知识图谱)**：检索目标疾病的精确定义、典型症状及常见并发症，构建疾病本体认知。
- **ClinicalTrials.gov**：提取历史成功临床试验的特征向量 (Embedding successful clinical trials)，通过语义相似度计算，召回Top相关的历史试验信息作为参考基准。
- **文献数据库 (Literature)**：通过相似度计算与重排 (Reranking) 算法，精准召回Top-5相关度最高的医学期刊摘要，提供最新研究证据。
- **DrugBank**：针对干预药物，检索其作用机制 (Mechanism of Action)、药代动力学 (Pharmacokinetics) 与代谢路径 (Metabolism)，确保用药安全性与科学性。

## 3.2 任务规划与图增强 Expert agent 执行 (Task Planning & Graph-Augmented Expert agents)

由于临床试验设计具有高度的复杂性，本研究引入了“分而治之”的Agentic Workflow。

首先，**任务规划器 (Task Planner)** 以临床试验监管指南 (Regulatory Guidelines) 为全局上下文环境，将整体宏大的设计目标分解为多个高度聚焦的、**针对当前试验定制的**子任务 (Subtasks 1, 2, 3…)；子任务的个数与划分由模型根据 trial configuration 自行决定，**而非**预设固定的专家角色模板。五维 review rubric 仅用于后续评审与优化，**不**规定 planner 的分解方式。

为提升智能体处理复杂关系的准确性，系统构建了**多维关系图谱 (Multi-Dimensional Graph)**，将多域RAG获取的异构信息进行结构化映射。在执行阶段，系统针对每个子任务分配一个专属的 **Expert agent（专家智能体）**。创新性地，系统采用**智能体子图提取技术 (Agentic Subgraph Extraction)**，为每个 **Expert agent** 精准切分并注入与其任务高度相关的知识子图。这不仅有效降低了LLM的上下文噪声，还大幅提升了子任务推理的逻辑严密性。

### 3.2.1 多维图谱构建与动态子图抽取机制 (Multi-Dimensional Graph Construction & Agentic Subgraph Extraction)

在处理复杂的临床试验设计任务时，直接将多域 RAG 召回的海量异构文本一次性灌入大语言模型（LLM），极易引发「上下文迷失 (Lost in the Middle)」与跨段语义难以组织。为解决这一痛点，本研究在检索碎片与专家推理之间插入一层 **会话级概念关系图**：用 **拓扑结构 + 边上的认知层级标签** 组织证据，并以 **渐进披露** 控制 token。

#### 	1. 多维关系图谱构建 (Multi-Dimensional Graph Construction)

系统汇总单次会话内多域 RAG 得到的碎片化文本（如 PrimeKG、ClinicalTrials.gov 相似试验、文献摘要、DrugBank 等），先行 **细粒度分块** 并建立 **向量索引**。随后以 **分批行格式 LLM 调用** 建图：**每一轮仅将下一批短文块** 送入模型，令其输出纯文本行 `*E1 (phrase)* relation *E2 (phrase)* <Extracted|Inferred>`，由代码解析并 **跨批合并**（同义 relation 折叠、边 **`frequency`** 计数）：

- **节点 (Nodes)**：可读 **`name`** + 括号简释 **`category`**；同名归一后由代码写入稳定 **`id`**。
- **边 (Edges)**：自由表述 **`relation`**（纯英文短语，不用 `<>` 包裹）；**`kind`** 来自行尾标签（**`extracted`** = 当批字面证据，**`inferred`** = 当批跨段合理关联，且不得违背临床医学常识）；**`frequency`** 为跨批/同义累计出现次数，**仅当 > 1 时写入边对象**（默认为 1 时省略）。不设 span 级对齐字段。
- **解析**：代码对 LLM 输出做 **严格行正则 + 单行 fallback**（去列表符、剥 relation 的 `<>`、取最后一个证据标签等）；**不做**整批 LLM 重试。

对每个节点 \(v\)，构造检索查询 **`name_v` 与一阶邻边短语（含 relation / kind）**，在向量库中取 **全局 Top‑1** chunk，记录 **`chunk_ref → chunk_id`**；多节点共享同一 chunk 时 **仅存引用**，附录投喂时 **按 `chunk_id` 去重**。

此处 **「多维」** 指 **多域证据汇入同一图**、**边上多元关系表述** 以及 **`extracted`/`inferred` 二分**。

#### 	2. 智能体驱动的子图提取 (Agentic Subgraph Extraction)

会话级图谱体量可控；面向某一子任务时，仍应避免把 **全文或未剪裁的结构** 直接交给所有下游模块。系统采用 **Agentic Subgraph Extraction**（图中常以紫色虚线表示 **仅连通「建图产物」与各 Expert agent**），包含三步渐进披露：

- **L0 目录**：**Expert agent** 收到 **全图节点的 `id` 与 `name` 列表**，无边、无正文 chunk，用于与子任务对齐并「点菜」选型。
- **L1 子图拓扑**：**Expert agent** 对 **`selected_ids` 先去重**，服务端在无向视图下取 **已选节点及其 1‑hop 邻居** 诱导 **`V', E'`**，并投喂 **节点摘要与边（含 relation 与 extracted/inferred）**；仍不附 chunk 正文。
- **L2 去重正文**：对 **`V'`** 中节点的 **`chunk_ref`** 取 **`unique(chunk_id)`** 后展开文本附录；顺序 **先 L1 后 L2**，以免在未建立结构前淹没于 raw。

默认 **不设多跳随机游走（RWR）或通用 K‑hop 扩散**；剪裁关键在于目录阶段的 **Expert agent** 选型 **与** **1‑hop** 邻居补齐。

#### 	3. Expert agent 赋能 (Expert agent augmentation)

每个负责特定子任务的 **Expert agent（专家智能体）** 经由上述 **L0→L2** 读取 **与其选型一致的子图拓扑与去重证据**，构成「精准投喂」，并以 **自然语言** 产出该 subtask 的结论（generation 阶段不强制 JSON 等结构化格式）。相较直接拼接原始检索段落，该设计在压缩上下文的同时保留 **显式结构与字面/推断边的可读区分**；Expert 产出再与 Trial Configuration、**撰写协议 (Writing Protocol)** 一并交给 **标准写手 (Criteria Writer)**，而 **Task Planner / Criteria Writer / Criteria Reviewer 不直接挂载整图或未抽取的全量图谱上下文**。

## 3.3 由简至繁 Workflow 下的标准整合 (Least-to-Most Integration)

**Least-to-most** 指 **整个 CriteriaAgent pipeline 的系统级 workflow**：Task Planner **分解** 复杂 eligibility 设计 → 各 Expert agent 在 routed 子图证据上 **局部作答** → Criteria Writer **收拢** sub-answers 为完整纳入/排除标准。这是架构层面的 decompose-and-integrate，**不是** Criteria Writer 内部的逐层提示技巧；Writer prompt 中**无需**再叠加「先写 phenotype、再 layered feasibility」类分层指令。

在各 Expert agent 完成局部推理后，所有的 **自然语言** subtask 答案将连同 **简洁有力的 Writing Protocol** 一起，作为上下文输入给 **标准撰写智能体 (Criteria Writer)**。Protocol 要求：ClinicalTrials.gov 式 Inclusion/Exclusion bullet 列表，且 **每条 criterion 单一、清晰、site 可执行**，避免模糊、不可操作的表述；数值与量表阈值仅在 trial configuration 或 expert 证据支持时给出。Writer 据此 **整合** 各 expert 结论，产出逻辑一致的初版受试者纳入与排除标准。

## 3.4 大模型裁判评估与自适应优化闭环 (LLM-as-a-Judge Framework & Optimization Loop)

为保证生成标准的临床实用价值与严谨性，本框架创新性地引入了**LLM-as-a-Judge**评审机制，构建了撰写者与评审者的双智能体博弈与协作闭环。

**标准评审智能体 (Criteria Reviewer)** 根据严格的医学伦理与实操规范，对初版标准进行**五维评价体系 (5-Dimensional Criteria Scoring)**（Criteria reviewer.md）的量化打分与详细文本反馈 (Comments)：

1. **信号富集 (Signal Enrichment)**：评估标准能否有效筛选出对干预措施敏感的患者亚群。
2. **人群触达 (Population Reach)**：评估标准的严苛程度是否会过度限制招募，影响入组效率。
3. **数据可评估性 (Data Evaluability)**：考察患者特征是否有利于最终临床终点数据的收集与统计学验证。
4. **临床可行性 (Clinical Feasibility)**：评估相关检测和筛选步骤在实际临床环境中的可操作性。
5. **风险缓解 (Risk Mitigation)**：评估排除标准是否充分剔除了存在严重用药禁忌或安全隐患的脆弱人群。

**自适应优化闭环 (Optimization Loop)**：Criteria Reviewer 将多维评分与针对性修改意见反馈给 Criteria Writer。Writer 吸收反馈后对标准进行迭代修正。此博弈-优化循环将持续进行，直至所有维度的评分达到预设阈值或达到最大迭代次数。最终，系统输出经过严格验证的**最终试验标准及多维评分报告 (Final Trial Criteria & Scores)**。