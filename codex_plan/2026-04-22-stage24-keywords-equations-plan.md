# Stage 24 Keywords and Equation Numbering Repair Plan

## Summary

修复两个 Stage 24 成稿问题：Word 中重复出现 `Keywords`，以及 Word 公式没有可见编号、正文缺少对公式编号的引用。修复不放宽 `docx_page_limit: 10`，不改变 Stage 22/23 职责，只在 Stage 24 canonical markdown、LaTeX/Word 转换防御层和 DOCX 后处理层收口。

## Key Changes

- 在 `researchclaw/pipeline/stage_impls/_final_editorial_repair.py` 增加关键词去重逻辑：同一篇稿件只保留一个 `**Keywords:**` block，优先保留关键词更多、更具体的那一条。
- 在 `_prepare_docx_markdown()` 再加一层防御：即使 canonical markdown 里意外残留多个 keywords，Word 专用稿也只输出一个 `custom-style="Keywords"` 段落。
- 在 `researchclaw/templates/converter.py` 的 LaTeX abstract keyword 提取里同步去重，避免第二个 keyword block 被当成正文落进 abstract。
- 为 Stage 24 的 display equation 增加全局编号语义：按正文顺序生成 `Equation (1)`, `Equation (2)` 等引用，并把相邻说明规范成 `In Equation (N), ...`。
- 在 DOCX 后处理 `_postprocess_editorial_docx()` 中识别 pandoc 生成的 display math 段落，为每个公式追加右侧可见编号 `(N)`；不依赖 `\tag{}`。
- 不把 literal `(N)` 写进 canonical markdown，避免 PDF 中 LaTeX 自动编号和手写编号重复。

## Test Plan

- 新增测试：两个 `**Keywords:**` 连续出现在 Abstract 时，规范化后只剩一个，且保留更具体的 keyword list。
- 新增测试：`_prepare_docx_markdown()` 面对重复 keywords 时只生成一个 `custom-style="Keywords"` block。
- 新增测试：LaTeX converter 提取 abstract keywords 时删除重复 keyword block，不把第二个 keyword 渲染进 abstract 正文。
- 新增测试：两个 display equations 经 Stage 24 normalize 后，正文出现 `Equation (1)` 和 `Equation (2)` 引用，公式说明出现 `In Equation (N), ...`。
- 新增 DOCX 后处理测试：包含两个 `m:oMathPara` 的最小 docx，经 `_postprocess_editorial_docx()` 后出现 `(1)`、`(2)`，并记录 `equation_numbers_present: true`。
- 回归运行 Stage 24 相关定向测试和 converter keyword/math 相关测试。

## Assumptions

- 所有 display equations 都按出现顺序编号；如果某个公式保留编号，就必须在相邻正文中出现对应 `Equation (N)` 引用。
- PDF 侧继续依赖 LaTeX `equation` 环境自动编号；Word 侧通过 DOCX 后处理补可见编号。
- 这轮只修 keywords 和公式编号/引用，不处理 13 页压到 10 页的问题；去掉重复 keywords 会略微减少 Word 篇幅。
