# Stage 24 PDF/DOCX Content Sync And Layout Parity Plan

## Summary

- `paper_repaired.md` 继续作为 Stage 24 的唯一 canonical 内容源。
- PDF 和 DOCX 必须从同一份 canonical 内容导出，内容保持 100% 一致。
- 现有 DOCX 侧要求全部保留，不回退：公式居中、编号右对齐、长公式兼容、图注编号、页数限制、现有公式说明与关键词等内容收口规则继续有效。
- 这次要修的不是内容规则，而是导出职责边界：Stage 24 的 PDF 不再被 Word 侧版式思路牵着走，视觉风格收敛到 Stage 22 的 LaTeX 论文观感。

## Implementation Changes

- 收敛 Stage 24 的内容源与压缩原则：
  - `paper_repaired.md` 先经过 Stage 24 现有内容收口流程，包括引用裁剪、页数约束触发时的 canonical 压缩。
  - `docx_page_limit` 超限时，继续压缩 canonical markdown，然后同时重导 PDF 和 DOCX，保证内容一致。
  - `paper_repaired_docx.md` 只负责 DOCX 兼容和排版适配，不能反向成为 PDF 或 canonical 的内容源。

- 保留现有 Stage 24 内容层规则，不回退：
  - 不移除现有公式说明、展示公式编号、关键词补齐、页数限制、引用上限等 Stage 24 内容规则。
  - PDF 与 DOCX 的“内容一致”指 canonical 内容一致，不要求两个格式共享完全相同的换行、对齐、OMML 或 LaTeX 环境写法。

- 重构 Stage 24 PDF 导出链，使其与 Stage 22 统一：
  - 把 Stage 22 现有 LaTeX 导出、样式准备、图表资源准备、引用与编译前修复逻辑抽成共享 helper。
  - Stage 22 继续通过该 helper 生成 `paper.tex` 与 `paper.pdf`。
  - Stage 24 改为对 `paper_repaired.md` 调用同一 helper，生成 `paper_repaired.tex` 与 `paper_repaired.pdf`。
  - Stage 24 不再保留简化版单独 LaTeX 编译路径，避免 PDF 在模板、图表、引用、编译修复链上与 Stage 22 漂移。

- 明确 PDF 和 DOCX 的职责分层：
  - PDF 只吃 canonical markdown，并走 Stage 22 同款 LaTeX 论文导出链，目标是会议模板观感。
  - DOCX 只在 `paper_repaired_docx.md` 和 DOCX postprocess 中处理 Word 兼容事项，例如 OMML 公式布局、caption style、分页适配。
  - 不把 Word 的表格布局、段落 tab、caption 样式之类概念带回 LaTeX 渲染层。

- 补充导出可见性：
  - 在 `editorial_final_assessment.json` 增加状态，标记本次最终导出是否走的是 shared canonical 模式，以及是否发生过 canonical 页数压缩。
  - 保持现有 `docx_quality.json` 检查项不降级，继续验证 DOCX 的公式和图注质量。

- 同步收口 Stage 24 editorial prompt：
  - 保留现有内容层要求。
  - 额外强调 PDF 版式目标是正常 LaTeX 论文观感，不是接近 Word 的页面排版。

## Public Interfaces

- 不新增配置项，继续沿用：
  - `export.docx_page_limit`
  - `export.max_references`
  - `export.target_conference`
  - `export.submission_profile`
- 导出物命名不变：
  - `paper_repaired.md`
  - `paper_repaired.tex`
  - `paper_repaired.pdf`
  - `paper_repaired.docx`
  - `paper_repaired_docx.md`
- `editorial_final_assessment.json` 新增导出模式与共享压缩状态字段，用于确认 PDF 与 DOCX 是否来自同一份最终 canonical 内容。

## Test Plan

- 在 `tests/test_rc_executor.py` 增加回归测试，覆盖 `docx_page_limit` 超限时：
  - 压缩发生在 canonical markdown。
  - 压缩后 PDF 和 DOCX 都重新导出。
  - `paper_repaired_docx.md` 不参与反向写回 canonical。

- 增加 Stage 24 PDF 导出一致性测试，确认它调用的是与 Stage 22 同源的 LaTeX 导出 helper，而不是旧的简化编译分支。

- 增加 PDF 资源准备回归测试，确认 Stage 24 与 Stage 22 一样完成：
  - 模板样式文件准备
  - 图表资源复制与路径修正
  - 编译前缺失资源修复逻辑

- 保留并重跑现有 DOCX 回归测试，确保以下行为不回退：
  - 公式居中
  - 公式编号右对齐
  - 长公式兼容
  - 图注编号
  - 页数限制仍然生效

- 增加一条端到端 Stage 24 导出测试，断言最终 `paper_repaired.md`、`paper_repaired.pdf`、`paper_repaired.docx` 的内容来源一致，且 assessment 中能看出是否发生过 shared canonical compression。

## Assumptions And Defaults

- “内容完全一致”按同一份 canonical markdown 约束，不要求 PDF 与 DOCX 的公式内部实现形式一致；DOCX 可以是 OMML，PDF 可以是正常 LaTeX `equation` 或 `aligned`。
- 公式说明、编号、页数限制都属于正确内容约束，应保留并同步到 PDF，而不是删除。
- 本次不追求让 Stage 24 PDF 回退成 Stage 22 的逐行逐页复制品，只要求它恢复到 Stage 22 那种正常 LaTeX 论文观感，而不是呈现出 Word 化排版痕迹。
