# Stage 24 Rerun DOCX/PDF Compile And Caption Repair Plan

## Summary

- 这轮不再调整 Stage 24 zip 打包范围；`artifacts/rc-20260415-085845-0de33f` 的标题命名 zip 已复核通过，当前问题集中在 `paper_repaired.docx` 与 `paper_repaired.pdf`。
- `paper_repaired.md` 继续作为 Stage 24 的唯一 canonical 内容源，PDF 和 DOCX 仍然必须从同一份内容导出。
- 这轮要修的根因有五个：
  - DOCX 长公式被写成原始 `$$ ... \begin{aligned} ... $$` 文本，而不是 OMML 数学对象。
  - 可单行容纳的展示公式被过度提升成 multiline / `aligned`。
  - 表格标题在 canonical markdown 或 DOCX 导出链中丢失，导致 `Table N. ...` 不再可见。
  - Stage 24 的 LaTeX 编译验证没有稳定复用 Stage 22 的 TeX 环境，触发 `algorithmic.sty` 缺失并把 `paper_repaired.tex/pdf` 拉进 fallback 降级链。
  - Stage 24 的公式解释收口规则过度激进，生成了 `denotes a variable defined in the surrounding text` 一类伪解释，污染 canonical 内容。

## Implementation Changes

- 收紧 `researchclaw/pipeline/stage_impls/_final_editorial_repair.py` 的 canonical markdown 公式收口：
  - 停止自动发明通用符号解释。
  - 保留原稿已有、语义完整的公式解释段。
  - 允许规范化已有 `In Equation (n), ...` / `where ...` 结构，但不再为缺失解释的公式自动补一整段伪参数说明。
  - 保留正常的公式 lead-in 和编号引用，不回退公式编号能力。

- 重写 DOCX 侧展示公式归一化逻辑：
  - 只有真正超长、已经多行、或明确存在对齐关系的公式才提升为 multiline。
  - 像 `\mathcal{L}_{cw}(\theta)=...` 这类单行可容纳公式保持单块显示，不生成 `aligned`。
  - 对 multiline 公式输出 Pandoc 能稳定转成 OMML 的数学块形式，禁止把 `$$`, `\begin{aligned}`, `\end{aligned}` 作为普通文本写进 `paper_repaired.docx`。
  - 在 DOCX 后处理前增加数学块有效性检查；若某种 multiline 写法不能稳定转 OMML，就回退成更简单但可正确渲染的 display math，而不是保留原始 LaTeX 文本。

- 恢复 PDF 的 Stage 22 风格编译主路径，并修正 Stage 24 的编译环境漂移：
  - 让 Stage 24 的编译验证与最终导出都复用 Stage 22 同一套 TeX 可执行、PATH 和样式搜索路径。
  - 优先修复“Stage 24 用 `/usr/share/texlive/...`，Stage 22 用 `/usr/local/texlive/2026/...`”这一环境不一致问题。
  - 对模板包做按需加载：若生成的 TeX 中根本没有 `algorithm` / `algorithmic` 环境，就不要无条件注入这两个包，避免因无关缺包导致整篇论文掉进 fallback。
  - 当主编译失败时，不再把以 `% WARNING: Compilation failed` 开头的降级 TeX 误当成成功的 `paper_repaired.tex` 最终产物。

- 修正表格标题在 PDF 和 DOCX 两侧的同源保留：
  - canonical markdown 中已有的 `Table N. ...` / `**Table N. ...**` caption block 必须保留，不能被 Stage 24 prose 压缩或段落改写吞掉。
  - LaTeX 侧继续把该 caption 吸收到 `table` float。
  - DOCX 侧把 table caption 明确渲染成 `TableCaption` 样式段，并保证编号和标题都保留。
  - 反解 LaTeX table 环境到 `paper_repaired_docx.md` 时，优先保留原 caption，而不是退化成自动生成的泛化 caption。

- 保持现有内容约束不回退：
  - `docx_page_limit` 继续有效；若修复公式与 caption 后 DOCX 页数回涨，仍通过 canonical markdown 的受控压缩来回收页数。
  - 公式编号、公式居中、编号右对齐、图注编号、`Keywords`、PDF/DOCX 内容同源原则全部保留。
  - 不允许为了让 PDF 更像 Stage 22 而回退到不同内容；目标始终是“Stage 24 内容 + Stage 22 风格 PDF”。

## Public Interfaces

- 不新增用户配置项。
- `docx_quality.json` 补充更精确的检查结果：
  - `display_math_omml_ok`
  - `table_caption_numbering_ok`
  - 必要时增加 `latex_compiler_env_consistent_with_stage22`
- `editorial_final_assessment.json` 同步暴露这些导出质量信号，避免 `clean: true` 被误读成“DOCX/PDF 完全正确”。

## Test Plan

- 在 `tests/test_rc_executor.py` 增加 DOCX 单行公式回归：
  - 输入单行展示公式。
  - 断言 `paper_repaired_docx.md` 不生成 `aligned`。
  - 断言导出的 `document.xml` 含 `m:oMath` / `m:oMathPara`，不含原始 `$$` 文本。

- 增加 DOCX 长公式回归：
  - 输入 `sigma_{F_1}` 这类必须多行的公式。
  - 断言最终 DOCX 仍为 OMML 数学对象，而不是裸 `\begin{aligned}` 文本。

- 增加表格标题回归：
  - 输入带 `Table 1. Summary of evaluation datasets.` 的 markdown 表格。
  - 断言 LaTeX 输出中保留 `\caption{Summary of evaluation datasets.}`。
  - 断言 DOCX 中存在 `TableCaption` 段，且含 `Table 1.`。

- 增加 Stage 24 编译环境与 fallback 回归：
  - 断言 Stage 24 的共享导出链会沿用 Stage 22 的 TeX 环境信息。
  - 对不含算法环境的论文，断言不会因为 `algorithmic.sty` 缺失而失败。
  - 对成功导出的 Stage 24 TeX，断言文件头不再是 `% WARNING: Compilation failed`。

- 跑现有 Stage 24 单测和导出回归后，再对 `artifacts/rc-20260415-085845-0de33f` 从 `FINAL_EDITORIAL_REPAIR` 做一次真实复跑验证：
  - `paper_repaired.docx` 中公式 14 不再乱码。
  - 可单行公式不再被突兀换行。
  - `Table 1. ...` 等表格标题恢复。
  - `paper_repaired.pdf` 重新回到 Stage 22 风格的正常 LaTeX 观感，而不是 Word 化排版。

## Assumptions And Defaults

- 用户的硬约束保持不变：PDF 与 DOCX 内容必须 100% 一致，只允许排版不同。
- 正常、语义完整的公式解释段很重要，应保留；本轮禁止的只是自动发明错误符号说明，不是禁止公式解释本身。
- 这轮默认优先修复 Stage 24 的导出链和后处理，不改 Stage 22 的既有职责边界。
- zip 打包逻辑本轮只做回归检查，不继续扩展范围。
