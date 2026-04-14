# ResearchClaw 定制化补充说明

这份文档不是官方总览，也不是开发日志。

它的定位保持不变：作为 [RESEARCHCLAW_AGENTS.md](/isilon_ai_data/zhangyq/AutoResearchClaw/RESEARCHCLAW_AGENTS.md) 的补充，让后续新会话在压缩上下文后，仍能快速理解这个仓库相对上游的真实运行方式、关键定制点、交付物边界和排查顺序。

## 建议读取顺序

1. `RESEARCHCLAW_AGENTS.md`
2. `AGENTS_ADD.md`
3. 当前真实运行配置文件

说明：
- 不要默认把 `config.researchclaw.example.yaml` 当成真实入口。
- 真实运行以你当前命令传入的配置文件为准，例如 `config2.arc.yaml`。

## 当前分支一眼看懂

- 当前分支已经从上游 23-stage 扩成 **24-stage pipeline**，新增 `Stage 24: FINAL_EDITORIAL_REPAIR`。
- `Stage 22` 仍负责基础导出正确性：论文导出、图表复制、图注清理、LaTeX/PDF 生成、代码包生成。
- `Stage 24` 负责最终成稿修复：Codex 介入修论文、重编译、做版式审查，并额外导出 Word。
- `research.paper_title` 已成为正式配置项，可强制指定论文标题。
- 旧 `deliverables/` 仍是兼容交付目录，不再被 Stage 24 接管。
- 新增 `deliverables_stage24/`，它才是 **Stage 24 优先** 的最终稿交付目录。
- 当前主 LLM 路径是 `ACP + codex`，不是直接走 OpenAI 兼容 API。
- Stage 10 代码生成和实验 repair 当前仍优先 `OpenCode`，不是 `codex exec`。
- FigureAgent 的 Gemini REST base URL 当前支持环境变量覆盖，不再写死在代码里。

## 真实入口与运行方式

### 配置入口

- 当前应优先看你实际运行命令里的配置文件，例如 `config2.arc.yaml`。
- `config.researchclaw.example.yaml` 只是示例，不代表当前真实运行状态。

### 主 LLM

- 主 LLM 是 `ACP + codex`。
- 也就是：ResearchClaw 通过 `acpx` 连接本机 `codex` 会话。
- 这不是浏览器 ChatGPT，也不是简单填 `OPENAI_API_KEY` 的直连模式。

### 代码与实验修复

- Stage 10 代码生成当前优先走 `OpenCode Beast Mode`。
- 实验 repair 当前也优先走 `OpenCode`。
- 不要把 `codex exec` 当成当前默认代码链路；它之前试过，但在当前环境里不如 `OpenCode` 稳定。

### 环境适配类改动

这些不是 Stage 24 核心功能，但属于当前环境中已经验证有效的稳定性改动：

- `researchclaw/agents/figure_agent/nano_banana.py`
  - Gemini REST base URL 优先读取 `GEMINI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`
  - 未配置时回退官方 `https://generativelanguage.googleapis.com/v1beta/`
- `researchclaw/llm/acp_client.py`
  - ACP session 初始化 timeout 放宽
- `researchclaw/pipeline/_helpers.py`
  - `sandbox` 自动补依赖时，只对 `pip` 子进程清理代理变量
  - 不再强制 `PIP_CONFIG_FILE=/dev/null`

不要把这些改动误判成应回退的本地污染。

## 当前新增功能主线

这次未提交代码的核心新增点，基本可以归为 5 条主线：

1. 24-stage pipeline
2. 论文标题硬约束 `paper_title`
3. Stage 22 图文导出修复
4. Markdown → LaTeX / Word 转换链修复
5. Stage 24 专用交付目录与压缩包

### 1. Stage 24：FINAL_EDITORIAL_REPAIR

已经新增并打通：

- `researchclaw/pipeline/stages.py`
- `researchclaw/pipeline/contracts.py`
- `researchclaw/pipeline/executor.py`
- `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
- `researchclaw/config.py`

同时补了配套：

- `researchclaw/hitl/smart_pause.py`
- `researchclaw/hitl/summarizer.py`
- `researchclaw/skills/schema.py`
- 多组 Stage 24 相关测试

### 2. 论文标题硬约束

已经把“研究主题”和“论文标题”拆开：

- `research.topic`：研究主题
- `research.paper_title`：论文标题

当前行为：

- `paper_title` 为空：标题仍由模型生成
- `paper_title` 非空：Stage 17 / Stage 19 prompt 会把它作为硬约束写入

这不是导出后改标题，而是 prompt 级约束。

### 3. Stage 22 图文导出修复

核心目标是把 Stage 22 从“能导出”提升到“基础正确”：

- 按正文实际引用复制图片，不再只认 `fig_*.png`
- 已引用图片优先读取 Stage 14 metadata 作为 caption
- 不再默认暴露内部文件名式图注
- 清理导出层污染文本
- 以 **figure bundle** 而不是单独 image block 进行局部重排

### 4. Markdown → LaTeX / Word 转换链修复

这部分横跨：

- `researchclaw/templates/converter.py`
- `researchclaw/templates/conference.py`
- `researchclaw/templates/compiler.py`
- `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`

目标不是加花活，而是修论文成稿链路：

- LaTeX 章节层级与 `\maketitle` / `abstract` 顺序
- figure/table caption 合并
- abstract 中 media block 迁出
- bibliography 前清理浮动体
- 数学公式保护
- 样式文件缺失时的 fallback
- Word 专用中间稿与 `reference.docx` 模板导出

### 5. Stage 24 专用交付目录

已经新增：

- `deliverables_stage24/`
- 按论文标题命名的 zip 压缩包

并且逻辑已放在：

- `researchclaw/pipeline/runner.py`

这个新目录与旧 `deliverables/` 平行存在，不互相替代。

## Stage 14：后续消费入口

- 后续阶段应优先把 `stage-14/` 理解为 canonical best。
- 不要默认把最新的 `stage-14_v*` 当成最终分析入口。
- 若存在 `experiment_summary_best.json`、`analysis_best.md`，优先按 best 视角理解，而不是按“最新版本目录”理解。

## Stage 22：职责与必须保留的修复

一句话总结：

- `Stage 22` 负责“别出错”
- `Stage 24` 负责“变好看”

### Stage 22 当前职责

- 导出 `paper_final.md` / `paper.tex` / `paper.pdf`
- 复制正文实际引用的图表资源
- 导出 `references.bib`
- 生成 `code/`
- 生成旧 `deliverables/`

### Stage 22 当前保留的关键修复

- 不再只复制 `fig_*.png`
- 优先从 Stage 14 metadata 读取 caption
- 去掉 `Lessons from Prior Runs`、`Learned Skills from Prior Runs`、`NeurIPS Paper Checklist`
- 去掉 `Figure N.` placeholder 和 prompt 式图注尾句
- 清理孤儿 markdown heading
- 去掉 framework diagram placeholder
- 用 figure bundle 思维做局部重排，而不是只挪图片块

### 为什么不能把这块全交给 Stage 24

因为 Stage 24 是终稿编辑器，不是资源装配器。

如果 Stage 22 不先把基础导出做对，Stage 24 的输入会先天变差：

- 图可能没复制对
- caption 可能还是内部文件名
- 正文可能残留导出污染段落
- 图、解释、caption 可能从一开始就是拆开的

这会直接抬高 Stage 24 的修稿成本，且显著降低稳定性。

### 当前 Stage 22 图处理的边界

- 不再激进地把大量未引用图片硬塞进正文
- 如需补图，只做少量、有 explanation 的补图
- 补图不能落进 abstract
- 方法主图优先回方法说明附近
- 流程图优先回 setup / protocol / experiments 附近
- 当前重排以局部高确定性修复为主，不做全局章节级乱排

## Stage 24：FINAL_EDITORIAL_REPAIR

### 定位

`Stage 24: FINAL_EDITORIAL_REPAIR` 位于 `CITATION_VERIFY` 之后。

它的职责不是替代 Stage 22/23，而是：

- 基于已有成稿继续修
- 让最终 PDF / DOCX / Markdown 更像可以直接交付的终稿

### 当前输入优先级

重跑 `FINAL_EDITORIAL_REPAIR` 时，输入优先级是：

1. `stage-24/paper_repaired.md`
2. `stage-23/paper_final_verified.md`
3. `stage-22/paper_final.md`

也就是说：

- 如果已经有 `stage-24` 成果，默认是续修
- 不是每次都从 22/23 重新开始

### 当前执行方式

- Stage 24 主路径是本地 Codex 介入修稿
- 工作区在 `stage-24/codex_repair_workspace/`
- 会多轮修改 `paper_repaired.md`
- 每轮后重新生成 repaired 版 LaTeX/PDF/Word
- 当前支持 `editorial_repair.mode`

### 当前 hard-fail 语义

以下情况会直接让 Stage 24 失败：

- 本地 `codex` CLI 不可用
- 修稿超时
- 多轮后仍未达到通过态
- citation key 被改坏

失败时不会把坏 repaired 版悄悄混进最终交付。

### 当前 boundary check 口径

当前 Stage 24 的 boundary check 已收缩为：

- 只保护 `citation key` 不变

默认不再单独阻断：

- 正文数字变化
- Figure/Table 编号变化
- 局部 prose 调整

### 当前输出

Stage 24 核心产物包括：

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

## Stage 24：PDF 布局与 Word 导出

### PDF 侧

Stage 24 不只看 Markdown 邻近性，也会看编译后的布局问题。

当前已经有 post-compile layout audit，会把明显问题记录为结构化 issue，例如：

- `awkward_float_layout`

它会补充更具体的信息：

- `target_kind`
- `target_hint`
- `section`
- `note`

后续不要再把这类问题简单理解成“图离引用太远”。

### Word 侧

Stage 24 当前已内置 Word 导出链：

- 从 `paper_repaired.md` 额外生成 `paper_repaired.docx`
- 使用 `paper_repaired_docx.md` 作为 Word 专用中间稿
- 使用模板 `researchclaw/templates/styles/reference.docx`

当前 Word 导出的关键事实：

- 目标是“可编辑论文稿”，不是简单 markdown dump
- 公式优先是 OMML 原生可编辑格式
- 不是 MathType OLE 对象
- 会尽量使用 citeproc 改善引用和参考文献样式

## Markdown → LaTeX 转换层：当前关键修复

这些不都写在 Stage 24 文件里，但它们属于当前成稿基础设施：

- `\maketitle` 与 `abstract` 顺序修正
- `# / ##` 混用时主章节层级错乱修正
- 图片前后 `Figure N.` 说明吸收入同一个 figure caption
- table caption 吸收入 table float
- abstract 中图片迁移到 abstract 后
- bibliography 前 `\clearpage`
- display math / inline math 的下标与求和保护
- 缺失 style 文件时的 fallback：
  - `geometry` margin fallback
  - numeric citation fallback

这些都属于应保留的新能力，不应再被当成“随手 patch”。

## 标题强制链路

当前 `research.paper_title` 已接入：

- `researchclaw/config.py`
- `config.researchclaw.example.yaml`
- `researchclaw/pipeline/stage_impls/_paper_writing.py`
- `researchclaw/pipeline/stage_impls/_review_publish.py`

当前行为：

- `paper_title` 为空：模型自己写标题
- `paper_title` 非空：写作 prompt 会要求使用完全一致的标题

这条链路是正式功能，不是本地手工覆盖。

## deliverables 与 deliverables_stage24 的边界

这是后续最容易混淆的地方。

### 旧 `deliverables/`

`deliverables/` 仍然是 pipeline 默认的兼容交付目录。

当前应理解为：

- 主要沿用 22/23 逻辑
- 不再被 24 步全面接管
- 不自动生成 zip

它不是 Stage 24 的原样快照。

### 新 `deliverables_stage24/`

`deliverables_stage24/` 是新增的 **24 步最终稿优先交付目录**。

它和旧 `deliverables/` 尽量保持同构，但内容优先来自 Stage 24：

- `paper_final.md` <- `stage-24/paper_repaired.md`
- `paper_final.docx` <- `stage-24/paper_repaired.docx`
- `paper.tex` <- `stage-24/paper_repaired.tex`
- `paper.pdf` <- `stage-24/paper_repaired.pdf`
- `references.bib` <- `stage-24/references.bib`
- `charts/` <- `stage-24/charts`

同时会补一些非 24 步但交付需要的内容：

- `code/` <- `stage-22/code`
- `verification_report.json` <- `stage-23/verification_report.json`
- `sanitization_report.json` <- `stage-22/sanitization_report.json`
- `.sty` / `.bst` <- `stage-22`

### Stage 24 专用压缩包

当前会自动生成一个按**论文标题命名**的 zip 压缩包。

这个 zip 对应的是 `deliverables_stage24/`，不是旧 `deliverables/`。

## report 与可见性变化

`researchclaw/report.py` 当前已经优先展示：

- `stage-24/paper_repaired.md`
- `stage-24/paper_repaired.tex`

如果 Stage 24 不存在，再回退展示 Stage 22 成果。

这意味着：

- `report` 层已经把 Stage 24 视为最终论文优先来源
- 但默认 `deliverables/` 没有被 Stage 24 接管

这两个判断不要混在一起。

## 数据与数据集：当前必须说清楚的事实

- 当前官方 `deliverables/` 和 `deliverables_stage24/` 都没有实验数据集打包逻辑
- 当前 artifact 中不存在独立 `data/` / `dataset/` 交付目录
- 当前实验代码更接近“代码内置/生成式输入”，不是“自动下载并打包完整数据集”

所以后续沟通时：

- 不要把 run 里的 JSON / report / summary 误称为“数据源包”
- 也不要默认认为交付目录会自动带数据集

## 哪些改动当前不建议回退

基于当前代码状态，下面这些改动都不建议为了“贴近主仓库”而盲目回退：

- Stage 24 的注册、配置、执行链路
- `research.paper_title`
- Stage 22 的 metadata caption / figure bundle / export cleanup 修复
- Markdown → LaTeX 的图表/章节/abstract/math 修复
- style 缺失时的 PDF fallback 修复
- Word 导出链与 `reference.docx`
- `deliverables_stage24/` 与标题命名 zip
- `nano_banana.py` 的 Gemini base URL 环境变量覆盖逻辑
- ACP timeout 放宽
- sandbox 自动补依赖时仅对 pip 清代理

## 当前最重要的判断原则

后续接手时，先按下面顺序判断，不要反过来：

1. 先区分问题发生在 Stage 22 还是 Stage 24
2. 先区分问题发生在 Markdown 层、converter 层还是编译后布局层
3. 先区分你看到的是旧 `deliverables/` 还是新的 `deliverables_stage24/`
4. 遇到图问题，先看 `paper_final.md / paper_repaired.md` 中的局部 bundle 是否已经成立
5. 遇到表问题，优先怀疑 `converter.py` 的 table caption / float 合并，而不是先怪 Stage 24
6. 遇到 Word 问题，先看 `paper_repaired_docx.md` 与 `reference.docx`
7. 遇到标题不对，先看 `research.paper_title`
8. 遇到页边距或引用样式异常，先怀疑 `.sty` 缺失后的 fallback 路径

## 建议排查顺序

1. 先看当前真实运行配置文件，而不是示例配置
2. 确认 `codex` / `acpx` / `opencode` 当前是否可用
3. 确认当前目标是旧 `deliverables/` 还是 `deliverables_stage24/`
4. 若问题在 PDF：
   - 先看 `paper_repaired.md` 或 `paper_final.md`
   - 再看 `paper_repaired.tex` 或 `paper.tex`
   - 最后再看编译后布局 issue
5. 若问题在 Word：
   - 先看 `paper_repaired_docx.md`
   - 再看 `reference.docx`
6. 若问题是图：
   - 先看 Stage 14 metadata
   - 再看 Stage 22 bundle 重排
   - 最后再看 Stage 24 是否续修成功

## 一句话总结

这个分支当前不是“原版 ResearchClaw + 少量 patch”，而是一条已经形成稳定主线的定制分支：

- 主 LLM 走 `ACP + codex`
- 代码生成与实验 repair 优先 `OpenCode`
- Stage 22 负责基础导出正确性
- Stage 24 负责最终 editorial repair
- 标题支持硬约束
- Markdown → LaTeX / DOCX 转换链已经围绕论文终稿质量做过系统修复
- 旧 `deliverables/` 保持兼容
- 新 `deliverables_stage24/` 负责承接 24 步最终稿

后续所有判断，都应默认从这个前提出发。
