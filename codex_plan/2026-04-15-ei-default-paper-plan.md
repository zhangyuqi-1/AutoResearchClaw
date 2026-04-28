# 论文生成链路优化实施计划

> **面向后续执行者：** 本文档沉淀本轮需求边界、统一设计原则、实现拆分与验证方案。实现时应优先保证“单一基础终稿内容源”原则，不允许 LaTeX 与 Word 在内容层面分叉。

**目标：** 在保留当前默认生成模式的前提下，补齐公共成稿质量问题，并新增 EI 会议模式，使默认模式与 EI 模式都能稳定产出结构一致、内容一致、仅排版略有差异的 LaTeX / Word 论文。

**架构原则：** 最终只有一份基础终稿内容，即 canonical final markdown。LaTeX 与 Word 都从这份基础终稿派生；如果因为页数、参考文献、结构约束触发 Stage 24 调整，调整对象必须是基础终稿内容本身，而不是只改单独的 Word 或 LaTeX 转换层。

**技术范围：** `researchclaw/config.py`、Stage 17/19 写作 prompt 注入、Stage 24 editorial repair、`researchclaw/templates/converter.py`、Word 中间稿生成、参考文献裁剪、Word 页数审计、FigureAgent 图表可读性修复、回归测试。

---

## 一、需求边界

### 1. 公共优化项

以下优化默认模式和 EI 模式都必须生效：

1. 论文补充 `Keywords`。
2. 英文文献索引在 Word 中使用正常字号数字引用，不再使用上角标。
3. 所有展示公式必须编号。
4. 所有展示公式中的参数都必须在文章中有对应解释。
5. 公式前后的语句必须自然连贯，不能出现“半句 + 公式 + 半句”或多余标点悬挂。
6. 图表不能出现图例遮挡数据、注释或关键内容的情况。

### 2. EI 模式专属项

只有明确标注为“EI 会议定制”的内容才属于 EI 模式专属：

1. 删除 `Related Work` 独立章节。
2. 结果章节标题改为 `Results and Analysis`。
3. `Results and Analysis` 正常拆成 3-4 个子章节，包含消融和指标分析。
4. `Discussion` 与 `Limitations` 合并进 `Conclusion`。

### 3. 通用配置项

以下限制项不与模式强绑定，而是通用配置：

1. `docx_page_limit`
2. `max_references`

规则固定如下：

1. 默认值为 `0`，表示不限制。
2. 显式配置为非 `0` 时，按硬约束执行。
3. 这两个限制项对默认模式和 EI 模式都通用。
4. 页数限制的评估口径按 Word 结果执行。
5. 参考文献限制按最终正文实际引用和最终 bibliography 一起校验。

## 二、统一设计原则

### 1. 单一基础终稿内容源

本轮最重要的设计约束：

1. Stage 24 内必须产出一份 canonical final markdown，作为唯一基础终稿内容。
2. LaTeX 与 Word 都从这份 canonical 内容派生。
3. LaTeX 与 Word 在内容层面必须一致：
   - 章节结构一致
   - `Keywords` 一致
   - 公式位置一致
   - 公式解释一致
   - 参考文献选取一致
4. 允许存在的差异只限于格式转换和排版层，例如：
   - 引用渲染方式
   - Word 样式块
   - LaTeX 模板样式
   - 页边距、字体、图注样式

### 2. 不允许的分叉实现

以下实现方式明确禁止：

1. 只在 LaTeX 层补 `Keywords`，Word 不补。
2. 只在 Word 层补公式解释，LaTeX 不补。
3. 只因 Word 页数超限而局部删改 `paper_repaired_docx.md`，不回写基础终稿。
4. LaTeX 和 Word 各自独立做章节重组，导致正文结构不同。

### 3. 正确的数据流

正确的数据流必须是：

1. Stage 17 / 19 先尽量按目标结构生成高质量正文。
2. Stage 24 对最终论文做 canonical 内容规范化。
3. LaTeX 导出从 canonical 内容渲染。
4. `paper_repaired_docx.md` 从 canonical 内容机械派生，仅做 Word 兼容适配。
5. 页数或文献限制若触发 Stage 24 继续修稿，修的是 canonical 内容本身。

## 三、配置与接口调整

### 1. `ExportConfig` 新增字段

计划新增：

```python
@dataclass(frozen=True)
class ExportConfig:
    target_conference: str = "neurips_2025"
    authors: str = "Anonymous"
    bib_file: str = "references"
    submission_profile: str = "default"
    docx_page_limit: int = 0
    max_references: int = 0
```

### 2. 字段语义

1. `submission_profile`
   - `default`
   - `ei_conference`
2. `docx_page_limit`
   - `0` 表示不限制
   - `>0` 表示 Word 页数硬限制
3. `max_references`
   - `0` 表示不限制
   - `>0` 表示最终引用数和参考文献数的硬限制

### 3. 明确保留的旧行为

1. `target_conference` 继续只负责 LaTeX 模板。
2. 默认模式继续走现有论文结构，不自动启用 EI 结构约束。
3. LaTeX 当前数字引用方式视为正确，不改其引用风格。

## 四、实施拆分

### 任务 1：建立 canonical final markdown 规范化阶段

**责任：** 在 Stage 24 中新增或收束一层统一内容规范化步骤，保证最终只有一份基础终稿内容源。

**涉及文件：**

1. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
2. 可能需要抽出到新的 helper，但内容规范化入口必须由 Stage 24 主控

**要做的事：**

1. 在 Stage 24 明确一个 canonical 内容处理入口。
2. 统一在这个入口完成：
   - 插入 `Keywords`
   - 识别展示公式
   - 补公式解释段
   - 清理公式前后的断裂句式
   - 统一章节标题和章节合并逻辑
3. 后续 LaTeX 和 Word 导出都只能从该 canonical 内容继续。

**关键输出要求：**

1. canonical 内容本身已包含 `Keywords`。
2. canonical 内容本身已包含公式解释。
3. canonical 内容本身已不再存在“半句后直接接独立公式”的问题。

### 任务 2：公共成稿规则注入写作阶段

**责任：** 让 Stage 17 / 19 不只是事后修，而是前面就尽量按正确结构生成。

**涉及文件：**

1. `researchclaw/pipeline/stage_impls/_paper_writing.py`
2. `researchclaw/pipeline/stage_impls/_review_publish.py`

**要做的事：**

1. 为默认模式与 EI 模式都注入公共规则：
   - 摘要后必须有 `Keywords`
   - 展示公式必须单独成块
   - 行内数学不要误升级
   - 每个展示公式后要解释参数
   - 公式前后句子必须完整
2. 为 EI 模式额外注入结构约束：
   - 无 `Related Work`
   - `Results and Analysis`
   - `Discussion` / `Limitations` 合并到 `Conclusion`

### 任务 3：LaTeX 与 Word 共用同一套公式语义

**责任：** 把“公式编号、行内/展示区分、公式解释内容”上移为基础终稿规则，再分别映射到 LaTeX 与 Word。

**涉及文件：**

1. `researchclaw/templates/converter.py`
2. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`

**要做的事：**

1. 在 canonical 内容中先确定公式语义：
   - 所有展示公式统一视为需要编号
   - 行内数学保持行内
2. LaTeX 转换层只负责把这套语义正确渲染成带编号环境。
3. Word 中间稿生成层也必须保留相同语义：
   - 展示公式单独成块
   - 行内数学仍保持行内
   - 公式解释段紧跟公式

**注意：**

这里不是“Markdown → LaTeX 专属规则”，而是 LaTeX 与 Word 共用的基础内容规则。

### 任务 4：修正 Word 中间稿的引用样式，但不让其承担内容修稿职责

**责任：** `paper_repaired_docx.md` 继续存在，但只能做 Word 兼容适配。

**涉及文件：**

1. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`

**要做的事：**

1. `paper_repaired_docx.md` 必须由 canonical 内容派生。
2. 只允许做以下 Word 适配：
   - 普通数字引用 `[1]` / `[1, 2]`
   - Word 自定义样式块
   - Word 数学渲染兼容
   - 图注 / 表注样式兼容
3. 不允许在这里：
   - 单独增删章节
   - 单独补 `Keywords`
   - 单独补公式解释
   - 单独删正文压页数

### 任务 5：实现通用页数与参考文献硬约束

**责任：** 限制项是通用配置，但评估口径和触发链路要明确。

**涉及文件：**

1. `researchclaw/config.py`
2. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
3. 可能补充 helper

**要做的事：**

1. `max_references > 0` 时：
   - 最终正文引用数不得超过上限
   - bibliography 不得超过上限
   - 若超限，回到 canonical 内容降引，而不是只改转换输出
2. `docx_page_limit > 0` 时：
   - 生成 DOCX 后，使用 `soffice --headless` 转 PDF
   - 用转换后 PDF 页数作为 Word 页数校验口径
   - 若超限，继续修改 canonical 内容压缩篇幅
   - 达到最大迭代仍超限则 Stage 24 失败

### 任务 6：新增 `Keywords` 统一插入规则

**责任：** 让 LaTeX 与 Word 都自然继承同一个关键词块。

**要做的事：**

1. 从 `research.domains`、主题、标题中抽取 3-5 个关键词。
2. 放在 canonical 内容中摘要后的位置。
3. LaTeX 与 Word 都从 canonical 内容继承这段内容。

### 任务 7：图表可读性修复

**责任：** 解决“图例盖住图本身内容”这类问题，尤其是当前 Figure 4 样例。

**涉及文件：**

1. FigureAgent 图表评审逻辑
2. 相关 matplotlib 脚本生成与后处理链
3. 回归样例：`fig_semantic_vs_fusion_scatter`

**要做的事：**

1. 增加图表评审规则：
   - legend 外置优先
   - 标签避让
   - 不遮挡数据
2. 为 matplotlib 脚本型图表增加通用 relayout 修复能力。
3. 用当前散点图作为强制回归用例。

### 任务 8：测试补齐

**责任：** 为本次大量功能变更建立稳定回归边界。

**建议补充的测试类型：**

1. 配置解析测试
2. Stage 17 / 19 prompt 注入测试
3. canonical 内容规范化测试
4. LaTeX / DOCX 一致性测试
5. DOCX 引用样式测试
6. 公式解释测试
7. EI 模式结构测试
8. `docx_page_limit` 超限失败测试
9. `max_references` 降引测试
10. 图表 legend 遮挡回归测试

## 五、EI 模式最终结构口径

EI 模式固定结构：

1. `Introduction`
2. `Method`
3. `Experimental Setup`
4. `Results and Analysis`
5. `Conclusion`

强制规则：

1. 不允许独立 `Related Work`
2. 不允许独立 `Discussion`
3. 不允许独立 `Limitations`
4. `Results and Analysis` 需要拆 3-4 个子部分

## 六、验收标准

### 1. 默认模式验收

1. 默认模式最终论文补齐 `Keywords`。
2. 默认模式 Word 引用为正常字号数字引用。
3. 默认模式展示公式全部带编号。
4. 默认模式展示公式后存在参数解释。
5. 默认模式公式前后句子自然，不再出现你给出的那种断裂样例。
6. 默认模式 LaTeX 与 Word 内容结构一致。

### 2. EI 模式验收

1. EI 模式不含 `Related Work`。
2. EI 模式结果章节标题为 `Results and Analysis`。
3. EI 模式 `Discussion` / `Limitations` 已并入 `Conclusion`。
4. 显式配置 `docx_page_limit=10` 时，Word 页数超过 10 页必须触发继续修稿或最终失败。
5. 显式配置 `max_references=20` 时，最终参考文献不得超过 20 条。

### 3. 单一内容源验收

1. 如果 Stage 24 因页数或参考文献限制触发进一步修稿，LaTeX 与 Word 都要反映同一轮 canonical 内容变化。
2. 不允许出现“Word 达标但 LaTeX 仍是旧内容”或反过来的情况。

## 七、实施顺序建议

1. 先改配置层，补 `submission_profile`、`docx_page_limit`、`max_references`。
2. 再做 canonical 内容规范化入口。
3. 再补 Stage 17 / 19 prompt 注入。
4. 再改 LaTeX / Word 共用公式语义链。
5. 再改 Word 引用样式和 `Keywords` 派生。
6. 再接入页数与文献限制。
7. 最后修图表可读性和补回归测试。

## 八、明确假设

1. 关键词本轮先自动生成，不新增手工关键词配置。
2. 所有“需要编号”的口径是：所有展示公式编号，行内数学不编号。
3. Word 仍然沿用当前导出路径，不改成由 LaTeX 直接转 Word。
4. 虽然 Word 不是从 LaTeX 直接转出来，但语义上必须视为“同一 canonical 内容的另一种格式投影”。

## 九、精确文件落点

### 1. 必改主文件

后续实现优先围绕以下文件展开：

1. `researchclaw/config.py`
   - 新增 `submission_profile`、`docx_page_limit`、`max_references`
   - 更新配置解析与默认值
2. `config.researchclaw.example.yaml`
   - 补充示例配置项与中文注释
3. `researchclaw/pipeline/stage_impls/_paper_writing.py`
   - Stage 17 写作 prompt 注入公共规则与 EI 专属规则
4. `researchclaw/pipeline/stage_impls/_review_publish.py`
   - Stage 19 修订 prompt 注入
   - 如已有结果章节/页数/图表相关后处理，也要同步接入新规则
5. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
   - canonical 内容规范化入口
   - `paper_repaired_docx.md` 派生规则
   - `docx_quality.json` 扩展
   - Word 页数审计
   - 参考文献上限与最终约束收口
6. `researchclaw/templates/converter.py`
   - 展示公式编号渲染
   - 行内数学与展示数学区分
   - 忠实渲染 canonical 内容中的关键词与公式解释

### 2. 高概率需要联动的文件

1. `researchclaw/pipeline/runner.py`
   - 若 `deliverables_stage24/manifest.json` 需要带出新配置或新产物说明，则同步调整
2. `researchclaw/agents/figure_agent/orchestrator.py`
   - 若图表质量规则在编排层注入，则在此落地
3. `researchclaw/agents/figure_agent/codegen.py`
   - 若需要在图表脚本 prompt 中注入 legend 外置/避让约束，则在此落地
4. `researchclaw/agents/figure_agent/planner.py`
   - 若图表规划阶段就要区分“散点图/多系列图需避让”，则在此落地
5. `researchclaw/agents/figure_agent/style_config.py`
   - 若要补统一 matplotlib 样式模板，可考虑在此落地

### 3. 现有测试文件落点

本轮优先复用和扩展以下测试文件，而不是另起一套零散测试：

1. `tests/test_rc_config.py`
2. `tests/test_rc_executor.py`
3. `tests/test_rc_runner.py`
4. `tests/test_rc_templates.py`
5. `tests/test_figure_agent.py`

### 4. 图表回归样例

当前明确问题图的回归样例路径：

1. `artifacts/rc-20260409-184727-b50802/stage-14/charts/scripts/fig_semantic_vs_fusion_scatter.py`

注意：

1. 这个路径属于历史产物，不是主源码。
2. 实现时应优先把规则加进 FigureAgent 源码；该脚本只作为“问题复现和回归验证样例”。

## 十、建议施工阶段与提交切片

这次需求面很大，不建议一次性改完。推荐按下面 6 个切片推进，每个切片都要能独立验证并提交。

### 阶段 A：配置层与模式边界

**目标：** 先把 `default` 与 `ei_conference` 的边界、以及通用上限配置立住。

**修改内容：**

1. `researchclaw/config.py`
2. `config.researchclaw.example.yaml`
3. `tests/test_rc_config.py`

**完成标准：**

1. 新字段能正确解析。
2. 默认值不影响现有行为。
3. `submission_profile` 只改变结构约束，不影响通用限制项的启用条件。

**建议提交信息：**

`feat: add submission profile and generic export limits`

### 阶段 B：canonical 内容规范化骨架

**目标：** 在 Stage 24 先建立“单一基础终稿内容源”的骨架，不急着一次补完所有细节。

**修改内容：**

1. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
2. `tests/test_rc_executor.py`

**完成标准：**

1. Stage 24 内有明确的 canonical 内容规范化入口。
2. LaTeX 和 DOCX 都从该入口结果继续派生。
3. `paper_repaired_docx.md` 不再允许单独补章节结构。

**建议提交信息：**

`refactor: centralize stage24 canonical final markdown flow`

### 阶段 C：公共内容修复

**目标：** 先修默认模式和 EI 模式都共享的问题。

**修改内容：**

1. `researchclaw/pipeline/stage_impls/_paper_writing.py`
2. `researchclaw/pipeline/stage_impls/_review_publish.py`
3. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
4. `researchclaw/templates/converter.py`
5. `tests/test_rc_executor.py`

**完成标准：**

1. `Keywords` 出现在最终基础终稿中。
2. 展示公式编号、公式解释、公式前后句子修复在 canonical 内容层完成。
3. LaTeX 与 DOCX 都继承这些内容，不再各修各的。

**建议提交信息：**

`feat: normalize final paper keywords and equation semantics`

### 阶段 D：EI 模式结构约束

**目标：** 单独落 EI 专属结构规则，避免与公共优化混在一起难排查。

**修改内容：**

1. `researchclaw/pipeline/stage_impls/_paper_writing.py`
2. `researchclaw/pipeline/stage_impls/_review_publish.py`
3. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
4. `tests/test_rc_executor.py`

**完成标准：**

1. EI 模式无 `Related Work`。
2. EI 模式结果主章为 `Results and Analysis`。
3. `Discussion` / `Limitations` 并入 `Conclusion`。

**建议提交信息：**

`feat: add ei conference paper structure profile`

### 阶段 E：Word 页数与文献限制硬约束

**目标：** 把通用限制项真正接入 Stage 24 执行链。

**修改内容：**

1. `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
2. `researchclaw/pipeline/runner.py`
3. `tests/test_rc_executor.py`
4. 视情况补充 `tests/test_rc_runner.py`

**完成标准：**

1. `max_references` 会回写 canonical 内容降引。
2. `docx_page_limit` 用 Word 结果口径校验。
3. 超限最终失败时不会输出伪合规终稿。

**建议提交信息：**

`feat: enforce generic docx page and reference limits`

### 阶段 F：图表可读性修复与回归

**目标：** 单独处理 FigureAgent 可读性问题，避免和正文修稿耦合。

**修改内容：**

1. `researchclaw/agents/figure_agent/orchestrator.py`
2. `researchclaw/agents/figure_agent/codegen.py`
3. `researchclaw/agents/figure_agent/planner.py`
4. `researchclaw/agents/figure_agent/style_config.py`
5. `tests/test_figure_agent.py`

**完成标准：**

1. legend 遮挡规则被注入源码链路。
2. 当前散点图样例可作为回归样例复现和验证。
3. 生成图的脚本风格有统一可读性约束。

**建议提交信息：**

`fix: improve figure readability and legend placement`

## 十一、每阶段具体执行清单

### 阶段 A 执行清单

1. 在 `ExportConfig` dataclass 中新增 3 个字段。
2. 在配置解析处接入这 3 个字段。
3. 在示例 YAML 中增加说明。
4. 扩展 `tests/test_rc_config.py`：
   - 默认值测试
   - 显式配置测试
   - `default` / `ei_conference` 边界测试

### 阶段 B 执行清单

1. 在 `_final_editorial_repair.py` 中找出现有最终 markdown 输入、修稿、LaTeX 导出、DOCX 导出的主流程。
2. 在修稿成功后的第一时间引入 canonical 内容变量，作为后续唯一来源。
3. 重构 `_prepare_docx_markdown()` 的调用点，让其输入来自 canonical 内容，而不是其他旁路文本。
4. 扩展 `docx_quality.json`，加入：
   - 是否包含 `keywords`
   - 是否存在普通数字引用
   - 是否检测到展示公式解释

### 阶段 C 执行清单

1. 在 Stage 17 / 19 prompt 中增加公共规则块。
2. 在 Stage 24 canonical 规范化中增加：
   - `Keywords` 插入
   - 展示公式与行内数学区分
   - 参数解释段补全
   - 悬挂标点清理
3. 在 `converter.py` 中确认展示公式最终渲染为带编号环境。
4. 在 `_prepare_docx_markdown()` 中确保 Word 继承同样的公式块与解释段。
5. 扩展 `tests/test_rc_executor.py` 对 `_prepare_docx_markdown()` 和 Stage 24 产物的断言。

### 阶段 D 执行清单

1. 在 Stage 17 / 19 prompt 中区分 `submission_profile`。
2. 在 Stage 24 规范化中加入 EI 结构收口逻辑。
3. 对已有 `paper_repaired.md` 结构回归样例构造测试：
   - 输入带 `Related Work`
   - 输出被合并为 EI 结构

### 阶段 E 执行清单

1. 在 Stage 24 的 DOCX 产出后增加 `soffice --headless` 页数审计。
2. 审计结果写入 `docx_quality.json` 或新的结构化字段。
3. 在参考文献收口阶段实现：
   - 按正文实际引用集合裁 bibliograpy
   - 若仍超限，则继续重写 canonical 内容
4. 为 Stage 24 的“超限失败”补测试，确保失败时不打包坏终稿。

### 阶段 F 执行清单

1. 找出 FigureAgent 中生成 matplotlib prompt 的位置。
2. 把 legend 外置、标签避让、避免覆盖数据加入 prompt 或样式模板。
3. 用当前 `fig_semantic_vs_fusion_scatter.py` 问题图复现遮挡。
4. 形成稳定回归测试或至少稳定回归 fixture。

## 十二、推荐验证命令

以下命令不是强制唯一方案，但建议按切片逐步执行，避免一次跑大而慢的全量测试。

### 配置层

```bash
pytest tests/test_rc_config.py -q
```

### Stage 24 / Word / canonical 相关

```bash
pytest tests/test_rc_executor.py -q
```

如果 `tests/test_rc_executor.py` 过大，优先按关键关键字过滤：

```bash
pytest tests/test_rc_executor.py -k "docx or stage24 or prepare_docx_markdown" -q
```

### Runner / 交付打包相关

```bash
pytest tests/test_rc_runner.py -q
```

### 模板 / LaTeX 转换相关

```bash
pytest tests/test_rc_templates.py -q
```

### FigureAgent 相关

```bash
pytest tests/test_figure_agent.py -q
```

## 十三、后续执行时的注意事项

1. 不要一上来就同时改 Stage 17/19、Stage 24、converter、FigureAgent 全部逻辑，先按切片推进。
2. 每完成一个切片，都先确认没有把默认模式的现有行为打坏。
3. 遇到页数超限或文献超限时，优先检查是否真的回写了 canonical 内容，而不是只动了 `paper_repaired_docx.md`。
4. Word 与 LaTeX 若出现内容差异，先回到 canonical 内容检查，而不是继续在转换层打补丁。
5. 图表问题优先在 FigureAgent 源码规则层修，不要依赖手工改历史产物脚本。
