# Stage 24 DOCX Repair And Compile Stabilization Plan

## Summary

- Scope 锁定为 `DOCX优先`。
- 先修 Stage 24 的 Word 导出链，让展示公式在 DOCX 中真正居中、编号右对齐，同时恢复图注里的 `Figure N.` 编号。
- 本次日志里真正需要优先处理的根因有三个：DOCX 后处理没有检查公式布局、DOCX 图注编号丢失、Stage 24 编译目录缺少会议样式文件导致 PDF 退化到 `article + geometry fallback`。
- 反复出现的 `bibtex ... couldn't open file name 'paper_repaired.aux'` 先按中间编译失败的伴随症状处理，不把它当成这轮首要修复目标。

## Implementation Changes

- 修改 `researchclaw/pipeline/stage_impls/_final_editorial_repair.py` 的 DOCX 后处理，让包含 `m:oMathPara` 的段落不再继承 `BodyText` 的首行缩进和两端对齐。
- 对展示公式段落强制写入“段落居中 + 零首行缩进 + 右侧 tab stop + 单个 `(N)` 编号”的布局规则，保留 OMML 可编辑性，不改成图片，也不改成表格。
- 在 DOCX 后处理里按文档顺序遍历 `CaptionedFigure` / `ImageCaption`，缺失编号时补成 `Figure N. Caption.`，已有编号时不重复追加。
- 不再依赖 Pandoc 对图片 alt-text 的隐式 caption 行为来保证图注编号；编号以 Stage 24 的 DOCX 后处理结果为准。
- 给 `docx_quality.json` 增加两个显式检查结果：`equation_alignment_ok` 和 `figure_caption_numbering_ok`。
- 把这两个新结果同步写入 `editorial_final_assessment.json`，让 Stage 24 能把这类 DOCX 回归暴露成结构化质量信号，而不是只靠人工打开文档发现。
- 在 Stage 24 的 LaTeX 编译前，按 Stage 22 的做法把当前会议模板对应的 `.sty/.bst/.cls` 复制到 `stage-24/`，避免继续退化到 fallback 版式。
- 这轮不重做 Stage 24 的 float 策略，只把 PDF 的明显根因收口到“恢复会议模板样式可用”。

## Public Interfaces

- `docx_quality.json` 新增 `equation_alignment_ok: bool`。
- `docx_quality.json` 新增 `figure_caption_numbering_ok: bool`。
- `editorial_final_assessment.json` 新增对应的 DOCX 评估字段。
- CLI、配置项、Stage 顺序不变。

## Test Plan

- 在 `tests/test_rc_executor.py` 增加 DOCX 后处理回归用例，覆盖“展示公式段落被居中、首行缩进被清掉、编号只追加一次、编号停在右侧”。
- 在 `tests/test_rc_executor.py` 增加图注回归用例，覆盖“未编号图注自动补成 `Figure 1.`、`Figure 2.`，已编号图注不被重复补号”。
- 在 `tests/test_rc_executor.py` 增加 `docx_quality.json` 新字段断言，确保这两个回归能被质量检查捕获。
- 增加 Stage 24 编译前样式文件准备的回归测试，确保 `target_conference: neurips_2025` 时不会再因为缺少 `neurips_2025.sty` 退化到 fallback。
- 跑现有 Stage 24 相关单测后，再对当前 artifact 复跑 `FINAL_EDITORIAL_REPAIR` 做一次定向验证。

## Assumptions And Defaults

- DOCX 图注默认格式采用 `Figure N. Caption.`，不使用 Word 的 `SEQ` 域。
- 展示公式继续保留为可编辑的 OMML，不在这一轮切换成边框隐藏表格布局。
- 这轮不尝试把 Stage 24 的 PDF 全面回退成 Stage 22 的所有排版细节，只先恢复“会议模板样式存在且生效”这个前提。
- `docx_page_limit` 压页逻辑保持现状，但之后它不能再掩盖 DOCX 公式/图注格式回归。
