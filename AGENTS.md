# ResearchClaw 定制化补充说明

这份文档是 [RESEARCHCLAW_AGENTS.md](/isilon_ai_data/zhangyq/AutoResearchClaw/RESEARCHCLAW_AGENTS.md) 的本地补充，不是官方总览，也不是逐日开发日志。

它的目标是：让后续新会话在上下文被压缩后，仍能快速判断这个仓库相对上游的真实运行方式、关键定制点、交付物边界和排查顺序。

## 建议读取顺序

1. `RESEARCHCLAW_AGENTS.md`
2. `AGENTS.md`
3. 当前命令实际传入的配置文件，例如 `config2.arc.yaml`
4. 对应 run 目录下的 `stage-*` 产物和日志

注意：

- 不要默认把 `config.researchclaw.example.yaml` 当成真实运行入口；它只是配置模板。
- 当前真实运行以 CLI 命令中的 `--config` 为准。
- `Quick_Start.md` 已补充从 Stage 24 resume 的命令：`researchclaw run --config config2.arc.yaml --resume --from-stage FINAL_EDITORIAL_REPAIR --auto-approve`。
- 当前沉淀的计划文档都在 `codex_plan/`，其中 2026-04-21 到 2026-04-24 的 Stage 24 相关计划大多已经执行。

## 计划文档索引

下面这些计划文档用于追溯设计背景和任务拆分；本文正文已经吸收其中与当前实现一致的结论。若计划文档与当前代码或本文正文冲突，优先以当前代码和本文正文为准。

- [codex_plan/2026-04-15-ei-default-paper-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-15-ei-default-paper-plan.md)：默认论文风格、EI 会议风格、canonical markdown、页数/引用约束的总体设计。
- [codex_plan/2026-04-15-sandbox-dataset-hardening-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-15-sandbox-dataset-hardening-plan.md)：本地 sandbox 三阶段执行、`RC_DATA_DIR`、setup/main 职责边界、数据集替代回退拦截。
- [codex_plan/2026-04-21-stage24-export-repair-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-21-stage24-export-repair-plan.md)：Stage 24 引用限制、citation/BibTeX 同步、EI completeness、DOCX 页数检测。
- [codex_plan/2026-04-22-stage24-round2-repair-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-22-stage24-round2-repair-plan.md)：Stage 24 markdown integrity guard、图文审计、章节词数统计、页数失败归因。
- [codex_plan/2026-04-22-stage24-keywords-equations-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-22-stage24-keywords-equations-plan.md)：Keywords 去重、展示公式编号、DOCX 公式编号后处理。
- [codex_plan/2026-04-23-stage24-docx-repair-and-compile-stabilization-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-23-stage24-docx-repair-and-compile-stabilization-plan.md)：DOCX 公式居中、图注编号、Stage 24 样式文件和编译稳定性。
- [codex_plan/2026-04-23-stage24-pdf-docx-content-sync-and-layout-parity-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-23-stage24-pdf-docx-content-sync-and-layout-parity-plan.md)：PDF/DOCX 同源、Stage 24 PDF 复用 Stage 22 风格导出链、canonical 压缩原则。
- [codex_plan/2026-04-24-stage24-rerun-docx-pdf-compile-and-caption-repair-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-24-stage24-rerun-docx-pdf-compile-and-caption-repair-plan.md)：DOCX 裸 LaTeX 公式、过度 aligned、表题保留、TeX 环境漂移、伪公式解释修复。
- [codex_plan/2026-04-24-stage24-figure-table-reference-and-equation-prose-repair-plan.md](/isilon_ai_data/zhangyq/AutoResearchClaw/codex_plan/2026-04-24-stage24-figure-table-reference-and-equation-prose-repair-plan.md)：图表显式 `Figure N` / `Table N` 引用、caption 恢复、公式 prose 自然化。

## 当前分支速览

- 上游 23-stage pipeline 已扩展为 **24-stage pipeline**，新增 `Stage 24: FINAL_EDITORIAL_REPAIR`。
- `Stage 22` 负责基础导出正确性：论文 Markdown/LaTeX/PDF、图表资源、BibTeX、代码包和旧 `deliverables/`。
- `Stage 23` 负责 citation verification，现在还会去重 BibTeX、清理 orphan citations 和未引用条目。
- `Stage 24` 负责最终成稿修复：Codex 修稿、canonical markdown 收口、LaTeX/PDF 重导出、Word 导出和质量审查。
- `research.paper_title` 是正式配置项；非空时 Stage 17/19 prompt 要求标题完全一致。
- `export.submission_profile` 已接入正文结构规则；当前支持正常会议论文风格 `default` 和 EI 会议压缩风格 `ei_conference`。
- `export.docx_page_limit` 和 `export.max_references` 是 Stage 24 的硬约束；`0` 表示关闭。
- 旧 `deliverables/` 仍保留兼容语义，不由 Stage 24 全面接管。
- 新增 `deliverables_stage24/` 作为 Stage 24 优先的内部完整交付目录。
- 标题命名 zip 是 Stage 24 的 submission-only archive，不再完整镜像 `deliverables_stage24/`。
- 当前主 LLM 路径是 `ACP + codex`，不是浏览器 ChatGPT，也不是 OpenAI 兼容 API 直连。
- Stage 10 代码生成和实验 repair 当前仍优先 `OpenCode Beast Mode`，不是 `codex exec`。
- 本地 sandbox 已支持 `requirements.txt -> setup.py -> main.py` 三阶段执行。
- 代码生成、校验、执行和诊断链已经禁止“目标数据集加载失败后偷偷换成别的数据集”。

## 当前真实配置倾向

当前 `config2.arc.yaml` 已切到 tabular / imbalanced classification 方向：

- `project.name`: `prmvai2026-tabular-ai`
- `research.paper_title`: `An Explainable Feature Selection and Ensemble Learning Framework for Imbalanced Tabular Classification`
- `experiment.benchmark_agent.enabled`: `true`
- `experiment.sandbox.data_root`: `/isilon_ai_data/zhangyq/AutoResearchClaw/.sandbox_data`
- `experiment.opencode.timeout_sec`: `3600`
- `export.submission_profile`: `ei_conference`
- `export.docx_page_limit`: `10`
- `export.max_references`: `20`

`.sandbox_data/*` 已加入 `.gitignore`，不要把它当成应提交的数据目录。

## 投稿风格边界

`export.submission_profile` 管正文结构和写作约束，`export.target_conference` 管 LaTeX 模板族。不要把两者混为一谈：同样使用 `neurips_2025` 模板时，`submission_profile: default` 和 `submission_profile: ei_conference` 会生成不同的正文组织方式。

### `default`：正常会议论文风格

- 适用于 NeurIPS / ICLR / ICML 这类完整会议论文结构。
- 保留独立 `Related Work`，用于系统讨论相关工作、差异和定位。
- 主结果章节默认使用 `Results`。
- `Discussion` 和 `Limitations` 可以作为独立章节保留。
- completeness check 会期待常规论文骨架，包括 related work、discussion/limitations 等独立结构。
- 风格目标是完整论证和充分展开，不主动为了 EI 页数压缩合并章节。

### `ei_conference`：EI 会议压缩风格

- 适用于页数和参考文献数更紧的 EI 类会议投稿。
- 不保留独立 `Related Work`；必要的相关工作比较并入 `Introduction`、`Method` 或 `Results and Analysis`。
- 主结果章节必须使用 `Results and Analysis`。
- 不保留独立 `Discussion` 或 `Limitations`；相关内容合并到 `Conclusion`。
- Stage 17 / Stage 19 写作 prompt、Markdown normalization、LaTeX completeness check 和 Stage 24 收口都会按 EI 结构处理。
- 若配置了非零 `docx_page_limit` 或 `max_references`，Stage 24 会把它们作为硬约束，在 canonical markdown 上压缩或裁剪后同时重导 PDF/DOCX。

当前 `config2.arc.yaml` 使用的是 `ei_conference`，并配置了 `docx_page_limit: 10`、`max_references: 20`。如果后续要回到正常完整论文风格，应显式改成 `submission_profile: default`，不要只改 `target_conference`。

## 当前实现变更地图

### 配置与入口

- `researchclaw/config.py`
  - 新增 `SandboxConfig.network_policy / auto_install_deps / pip_timeout_sec / setup_timeout_sec / data_root`。
  - 新增 `ExportConfig.submission_profile / docx_page_limit / max_references`。
  - `validate_config()` 会拒绝非法 `export.submission_profile` 和负数或非整数导出限制。
- `config.researchclaw.example.yaml`
  - 补齐 sandbox 下载/安装字段。
  - 补齐 `export` 段及 EI/页数/引用限制说明。
- `Quick_Start.md`
  - 增加 resume 到 `FINAL_EDITORIAL_REPAIR` 的命令示例。

### ACP、OpenCode 与 FigureAgent

- `researchclaw/llm/acp_client.py`
  - ACP reconnect 现在识别 `queue owner disconnected`。
  - reconnect 判断改成小写匹配，避免大小写漏判。
- `researchclaw/pipeline/opencode_bridge.py`
  - `opencode run` 加 `--dangerously-skip-permissions`，用于非交互流水线写 workspace。
  - `_collect_files()` 会递归收集 Python 文件以及嵌套的 `setup.py` / `requirements.txt`。
  - 若 OpenCode 返回成功但没有 `main.py`，会写 `stage-10/opencode_attempt_*.log` 并保留 workspace 供排查。
- `researchclaw/agents/figure_agent/codegen.py`
  - line/scatter/bar 模板 legend 改为图外右侧，避免遮挡图内曲线或点。

### Sandbox、数据与实验校验

- `researchclaw/experiment/sandbox.py`
  - 本地 sandbox 执行顺序是 `pip install -r requirements.txt`、`python setup.py`、`python main.py`。
  - 会注入 `RC_DATA_DIR`、`HF_HOME`、`HF_CACHE`、`HF_DATASETS_CACHE`、`TRANSFORMERS_CACHE`、`SKLEARN_HOME`、`PIP_CACHE_DIR`。
  - 本地 sandbox 的 `network_policy` 只接受 `full` 和 `none`。
  - 若 `network_policy='none'` 且存在 `requirements.txt` 或 `setup.py`，直接失败，不伪装离线可跑。
  - pip/setup phase 会写 `phase-*.stdout.log` / `phase-*.stderr.log`；main phase 输出会并入最终 `SandboxResult`。
- `researchclaw/experiment/validator.py`
  - 新增 `validate_project_files()`，检查跨文件项目约束。
  - 根目录执行的项目不能在 `main.py` 等顶层脚本里使用 `from .xxx import ...`。
  - 可下载数据集工作流必须有 `setup.py`。
  - `setup.py` 必须使用 `os.environ['RC_DATA_DIR']`。
  - 禁止硬编码 `/workspace/data`。
  - 禁止 `load_breast_cancer`、`load_wine`、`using ... fallback for` 等替代数据集回退。
- `researchclaw/pipeline/stage_impls/_code_generation.py`
  - sandbox 模式现在读取 `config.experiment.sandbox.network_policy`，不再默认当作无网络。
  - Stage 10 会执行项目级校验；无 `setup.py` / `requirements.txt` 的平面项目还会做短超时 smoke check。
  - 生成的 experiment spec 不再写死 “no external data, no network”，而是描述配置化 setup/data workflow。
- `researchclaw/pipeline/stage_impls/_execution.py`
  - Stage 12 即使 exit code 为 0，只要输出包含数据集替代回退信号，也判 `failed`。
- `researchclaw/pipeline/experiment_diagnosis.py`
  - 新增 `DATASET_SUBSTITUTION` 和 `SETUP_PHASE_MISSING`。
  - 这两类 critical deficiency 会把论文模式压到 `TECHNICAL_REPORT`。
- `researchclaw/pipeline/experiment_repair.py` 和 `researchclaw/prompts.py`
  - repair prompt 会列出 expected datasets，并明确禁止换成 unrelated fallback。
  - 代码生成 prompt 要求 setup 阶段准备数据，`main.py` 只消费目标数据集。

### 写作、转换、编译与引用

- `researchclaw/pipeline/stage_impls/_paper_writing.py`
  - 新增 shared submission rules：摘要后 `Keywords`、展示公式独立成块、公式前 lead-in、公式后必要符号解释。
  - `ei_conference` 不生成独立 `Related Work`，结果章节用 `Results and Analysis`，`Discussion` / `Limitations` 并入 `Conclusion`。
- `researchclaw/templates/converter.py`
  - `markdown_to_latex()` 接收 `submission_profile`。
  - abstract 中的 `**Keywords:**` 会导出到 LaTeX abstract 后、正文前。
  - plain `Table N. ...` caption 也会识别并并入 table float。
  - EI profile 的 completeness check 不再要求独立 `Related Work` / `Limitations`。
  - section word count 会聚合子章节内容，避免主章节 body 为空时误报。
  - 未使用 algorithm 环境时会去掉 `algorithm` / `algorithmic` package，缓解缺包编译失败。
- `researchclaw/templates/compiler.py`
  - `pdflatex` / `bibtex` 优先从 `/usr/local/texlive/*/bin/*/` 解析，再回退 PATH。
  - 这是为避免 Stage 22/24 用到不同 TeX Live 环境。
- `researchclaw/pipeline/stage_impls/_review_publish.py`
  - 新增 `_export_latex_pdf_artifacts()`，Stage 24 用它复用 Stage 22 风格的 LaTeX 模板、样式复制、图表准备、missing figure preflight 和编译链。
  - Stage 22 原有导出路径与 Stage 24 helper 都会把 `submission_profile` 传给 `markdown_to_latex()`。
  - Citation Verify 会去重 BibTeX，优先 DOI-backed 非 arXiv 版本。
  - 引用裁剪会同时清理 markdown citation cluster、orphan citation key 和未引用 BibTeX entry。

### Stage 24 成稿修复

- `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
  - `paper_repaired.md` 是唯一 canonical final markdown。
  - `paper_repaired.tex`、`paper_repaired.pdf`、`paper_repaired_docx.md`、`paper_repaired.docx` 都必须从 canonical markdown 派生。
  - Stage 24 会补/去重 `Keywords`，按 profile 调整结构，恢复图表 caption 和显式引用，规范展示公式和必要符号解释。
  - 保留的图和表必须在附近正文中出现显式 `Figure N` / `Table N` 引用。
  - `_audit_markdown()` 会报 `missing_explicit_figure_reference` 和 `missing_explicit_table_reference`，严重度为 high。
  - 公式 prose 默认自然表达，不再强制生成 `in Equation (n)` 模板句。
  - 会过滤伪符号说明，例如 `denotes a variable defined in the surrounding text`、`quad denotes the probe function`、`arg/max/operatorname` 等。
  - Word 侧只对真正过长或天然多行的展示公式提升成 `aligned`，短公式保持单行。
  - DOCX 后处理使用“同段落 OMML 公式 + 居中 tab + 右对齐编号 tab”的路径，不再回到可见表格方案。
  - `docx_quality.json` 现在覆盖 `display_math_omml_ok`、`equation_alignment_ok`、`figure_caption_numbering_ok`、`table_caption_numbering_ok`、`keywords_present`、`numeric_citations_plain`、`reference_limit_ok`、`docx_page_limit_ok`。
  - Stage 24 最终状态要求 Codex loop 成功且 `docx_quality.json["issues"]` 为空；否则 hard fail。
  - DOCX 页数检查使用 `.docx_page_count_pdf/` 隔离目录，不能覆盖 `stage-24/paper_repaired.pdf`。
  - 若 `docx_page_limit` 超限，会压缩 canonical markdown 后同时重导 PDF 和 DOCX，不做 DOCX-only 删除。
  - `editorial_final_assessment.json` 会记录 shared canonical、页数压缩轮次、DOCX/PDF 质量字段和引用限制结果。

### Deliverables 与打包

- `researchclaw/pipeline/runner.py`
  - `deliverables_stage24/` 继续作为内部完整交付目录，优先拷贝 Stage 24 产物。
  - 样式文件拷贝范围扩展到 `.sty` / `.bst` / `.cls`。
  - 标题命名 zip 改为 submission-only archive，根目录直接放提交文件。
  - zip 只包含 `paper.tex`、`paper.pdf`、`paper_final.docx`、`references.bib`、模板样式、`charts/`、`code/`、`data/`。
  - zip 不包含 `paper_final.md`、manifest、review JSON、quality JSON、编译中间文件和 chart prompt markdown。
  - `charts/` 下 markdown/prompt 辅助文件会被过滤。
  - `data/` 只认 run 内本地产物；找不到可分发数据时写 `data/README.md`，提示用 `code/setup.py` 和 `RC_DATA_DIR` 复现。

## Stage 14 / 22 / 23 / 24 边界

### Stage 14

- 后续阶段应优先把 `stage-14/` 理解为 canonical best。
- 不要默认把最新的 `stage-14_v*` 当成最终分析入口。
- 若存在 `experiment_summary_best.json`、`analysis_best.md`，优先按 best 视角理解。

### Stage 22

`Stage 22` 的职责是“别出错”：

- 导出 `paper_final.md` / `paper.tex` / `paper.pdf`。
- 按正文实际引用复制图表资源，不再只认 `fig_*.png`。
- 优先从 Stage 14 metadata 读取 caption。
- 清理导出污染文本，例如 prior-run lessons、checklist、placeholder 图注、孤儿 heading、framework diagram placeholder。
- 以 figure bundle 做局部重排，而不是只挪单独 image block。
- 导出 `references.bib`、`code/` 和旧 `deliverables/`。

不要把 Stage 22 的资源装配能力全交给 Stage 24。Stage 24 是终稿编辑器，不是资源装配器；如果 Stage 22 输入先天错误，Stage 24 成本和失败率都会上升。

### Stage 23

`Stage 23` 负责 citation verification：

- 低相关引用会从正文和 BibTeX 同步移除。
- 多 key markdown citation cluster 会保留剩余 key，例如 `[a, b, c]` 删除 `b` 后变成 `[a, c]`。
- 验证后会清理 orphan citation key 和 uncited BibTeX entry。
- BibTeX key 重复时，优先保留 DOI-backed、非 arXiv/preprint 变体。

### Stage 24

`Stage 24: FINAL_EDITORIAL_REPAIR` 位于 `CITATION_VERIFY` 之后。

输入优先级：

1. `stage-24/paper_repaired.md`
2. `stage-23/paper_final_verified.md`
3. `stage-22/paper_final.md`

含义：

- 若已有 `stage-24` 成果，默认是续修。
- 若要验证新规则，不要保留旧 `stage-24` 后误以为从 Stage 23 重新开始。
- 想验证 FigureAgent + 写作 + Stage 24 全链路，至少从 Stage 14 后重跑。

核心输出：

- `paper_repaired.md`
- `paper_repaired.tex`
- `paper_repaired.pdf`
- `paper_repaired.docx`
- `paper_repaired_docx.md`
- `editorial_review.json`
- `editorial_iterations.json`
- `editorial_final_assessment.json`
- `codex_review.json`
- `docx_quality.json`

hard-fail 语义：

- 本地 `codex` CLI 不可用。
- 修稿超时。
- 多轮后仍未达到通过态。
- citation key 被改坏。
- markdown 结构损坏，例如未闭合 fenced code block、标题丢失、必需 heading 丢失。
- DOCX 质量检查存在 issue。

boundary check 当前只保护 citation key，不再单独阻断正文数字变化、Figure/Table 编号变化或局部 prose 调整。

## PDF / DOCX 同源原则

- `paper_repaired.md` 是 Stage 24 的唯一内容源。
- PDF 和 DOCX 必须来自同一份 canonical markdown。
- `paper_repaired_docx.md` 只允许做 Word 兼容适配，例如 OMML、caption style、长公式 reflow；不能成为新的内容源。
- 若页数或引用限制触发继续修稿，必须改 `paper_repaired.md`，然后同时重导 PDF 和 DOCX。
- Stage 24 的 PDF 走 `_review_publish._export_latex_pdf_artifacts()`，目标是正常 LaTeX 论文观感，不是 Word 页面效果。
- 若再次看到 Stage 24 PDF 像 Word，先运行 `pdfinfo stage-24/paper_repaired.pdf` 看 `Creator/Producer`；若是 LibreOffice，优先怀疑 DOCX 转 PDF 页数检查产物覆盖。

## Word 导出注意事项

- Word 目标是“可编辑论文稿”，不是 markdown dump。
- 展示公式应尽量是 OMML 原生可编辑对象，不是 MathType OLE，也不是裸 LaTeX 文本。
- 数字引用应为普通字号 `[1]` / `[1, 2]`，不再使用上角标。
- `reference.docx` 位于 `researchclaw/templates/styles/reference.docx`。
- `docx_quality.json clean: true` 只表示当前检查项通过；如果检查项不覆盖某类版式问题，不能把它等同于人工版式完全正确。
- 若 DOCX 中出现裸 `$$`、`\begin{aligned}` 或 `\sum`，优先查 `paper_repaired_docx.md` 和 `display_math_omml_ok`。
- 若图注或表题编号缺失，先看 canonical markdown 是否有 `Figure N` / `Table N` caption，再看 DOCX style 是否落到 `ImageCaption` / `TableCaption`。

## 数据与数据集原则

- 旧 `deliverables/` 不主动打包实验数据集。
- `deliverables_stage24/` 是内部完整交付/排查目录，也不主动镜像共享缓存。
- 标题命名 Stage 24 zip 有 `data/` 入口。
- 数据打包只认 run 内本地产物，例如 `stage-24/data`、`stage-24/dataset`、`stage-22/data`、`stage-22/dataset`、`stage-22/code/data`、`stage-22/code/dataset`。
- 不扫描共享 `.sandbox_data`、HF cache、sklearn cache 再偷偷塞进 zip。
- run 内没有真实数据目录时，zip 中写 `data/README.md`，说明用 `code/setup.py` 和 `RC_DATA_DIR` 复现。
- 可下载数据集的正确路径是 setup 阶段准备数据，main 阶段严格消费数据。
- `main.py` 不应联网下载，也不应在目标数据集失败后换小数据集保活。
- 不允许用 `breast_cancer` 替代 Adult，不允许用 `wine` 替代 Covtype。
- 如果目标数据集仍不可用，应 `RuntimeError`，不要 silent fallback。

## Deliverables 边界

### 旧 `deliverables/`

- 主要沿用 Stage 22/23 逻辑。
- 不自动生成标题命名 zip。
- 不是 Stage 24 的原样快照。

### 新 `deliverables_stage24/`

优先来自 Stage 24：

- `paper_final.md` <- `stage-24/paper_repaired.md`
- `paper_final.docx` <- `stage-24/paper_repaired.docx`
- `paper.tex` <- `stage-24/paper_repaired.tex`
- `paper.pdf` <- `stage-24/paper_repaired.pdf`
- `references.bib` <- `stage-24/references.bib`
- `charts/` <- `stage-24/charts`

同时补充排查需要的非 Stage 24 产物：

- `code/` <- `stage-22/code`
- `verification_report.json` <- `stage-23/verification_report.json`
- `sanitization_report.json` <- `stage-22/sanitization_report.json`
- `.sty` / `.bst` / `.cls` <- `stage-22`

### 标题命名 zip

- 这是 submission-only archive。
- 归档根目录直接放提交文件，不套 `deliverables_stage24/`。
- 不混入内部 review JSON、manifest、markdown 源稿或编译中间文件。
- 针对 `artifacts/rc-20260415-085845-0de33f` 的复核中，zip 边界已经通过；后续首要问题是成稿导出质量，不是 zip 结构。

## 不建议回退的改动

基于当前代码状态，下面这些不应为了“贴近主仓库”而盲目回退：

- Stage 24 注册、配置、执行链路。
- `research.paper_title` 硬约束。
- `export.submission_profile`、`docx_page_limit`、`max_references`。
- Stage 22 metadata caption、figure bundle、export cleanup 修复。
- Stage 23 citation/BibTeX 同步清理和去重。
- Stage 24 canonical markdown、PDF/DOCX 同源、DOCX 质量 hard-fail。
- Word 导出链、`reference.docx`、OMML 公式后处理。
- Stage 24 PDF 复用 Stage 22 LaTeX/PDF helper。
- `deliverables_stage24/` 与标题命名 submission-only zip。
- sandbox 三阶段执行、`RC_DATA_DIR`、共享 cache env、项目级数据集约束。
- OpenCode `--dangerously-skip-permissions`、递归收集 support files、失败 debug workspace。
- ACP reconnect 对 `queue owner disconnected` 的处理。
- FigureAgent legend 外置，避免图内遮挡。
- Gemini REST base URL 环境变量覆盖逻辑。
- sandbox 自动补依赖时仅对 pip 子进程清理代理。

## 排查顺序

先做全局判断：

1. 看当前真实运行配置文件，不要看示例配置下结论。
2. 确认当前目标是旧 `deliverables/`、新 `deliverables_stage24/`，还是标题命名 zip。
3. 确认问题发生在 Stage 22、23 还是 24。
4. 确认问题发生在 Markdown 层、converter 层、LaTeX 编译层、DOCX 后处理层还是打包层。

遇到实验数据问题：

1. 看 `experiment.sandbox.network_policy`。
2. 看生成代码是否包含 `setup.py`。
3. 看 `setup.py` 是否使用 `RC_DATA_DIR` / HF / sklearn cache env。
4. 看 `main.py` 是否只消费目标数据集。
5. 看 Stage 10 project validation、Stage 12 stdout/stderr、experiment diagnosis 中是否出现 `DATASET_SUBSTITUTION` 或 `SETUP_PHASE_MISSING`。

遇到 PDF 问题：

1. 看 `stage-24/paper_repaired.md` 或 `stage-22/paper_final.md`。
2. 看 `paper_repaired.tex` 或 `paper.tex`。
3. 确认 Stage 24 是否走 `_review_publish._export_latex_pdf_artifacts()`。
4. 看 TeX binary 是否来自 `/usr/local/texlive` 预期路径。
5. 若 PDF 像 Word，先查 `pdfinfo` 的 `Creator/Producer`。
6. 最后看 compiled layout audit issue，例如 `awkward_float_layout`。

遇到 Word 问题：

1. 看 `stage-24/paper_repaired_docx.md`。
2. 看 `reference.docx` 是否存在。
3. 看 `docx_quality.json` 的具体 issue，而不是只看 `clean`。
4. 公式问题先查 `display_math_omml_ok` 和 `equation_alignment_ok`。
5. 图表题问题先查 canonical markdown 中的 caption 与显式引用，再查 DOCX style。

遇到图表问题：

1. 先看 Stage 14 metadata。
2. 再看 Stage 22 `paper_final.md` 中局部 figure bundle 是否成立。
3. 再看 Stage 24 是否恢复 caption 和显式 `Figure N` / `Table N` 正文引用。
4. 不要用 `the figure below` / `the table below` 替代显式编号引用。

遇到 EI 结构、超页或超引问题：

1. EI 结构先看 `export.submission_profile`。
2. 超页先看 Stage 24 是否回写 canonical `paper_repaired.md` 并同时重导 PDF/DOCX。
3. 超引先看 `_enforce_reference_limit()` 后 markdown 与 `references.bib` 是否同步。
4. 不要只在 DOCX 中间稿单独删正文或删引用。

## 复测旧 artifact

如果要对已有 run 目录复测，例如 `artifacts/rc-20260415-085845-0de33f`：

- Stage 24 输入优先级会先吃旧 `stage-24/paper_repaired.md`。
- 只想验证 Stage 24 新规则时，至少移除旧 `stage-24`。
- 想验证写作和 FigureAgent 到 Stage 24 的全链路时，从 Stage 14 后重跑更稳。
- 想验证 citation 到 Stage 24 的链路时，至少从 Stage 23/24 交界重新跑。

## 一句话总结

这个分支不是“原版 ResearchClaw + 少量 patch”，而是已形成稳定主线的定制分支：主 LLM 走 `ACP + codex`，代码生成和实验 repair 优先 `OpenCode`，sandbox 支持 setup/data workflow，Stage 22 负责基础导出，Stage 23 负责引用清理，Stage 24 负责 canonical 终稿修复与 PDF/DOCX 同源交付，最终提交以 Stage 24 标题命名 zip 为准。
