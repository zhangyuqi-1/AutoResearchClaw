# Stage 22-24 Export Repair Plan

## Summary

修复当前 run 的主故障链，目标是让 Stage 24 在 `ei_conference` 模式下稳定产出结构正常、引文一致、BibTeX 可编译、DOCX 页数可判定的终稿，并避免 deliverables 再靠“事后剥 orphan cite”兜底。

## Key Changes

- 修 `Stage 24` 的 reference-limit 后处理。
  - 修改 `researchclaw/pipeline/stage_impls/_final_editorial_repair.py` 的 `_enforce_reference_limit()`，只裁剪 citation cluster，不再做全局 `\\s{2,}` 压缩。
  - 保留段落空行、标题行、表格行和图片 bundle 的原始换行。
  - 在 `_normalize_final_paper_markdown()` 之后增加结构自检：至少要还能解析出标题、`Abstract`、`Introduction`，否则 Stage 24 直接失败并回退到未损坏 markdown。

- 让 completeness check 感知 `submission_profile`。
  - 修改 `researchclaw/templates/converter.py`，给 `check_paper_completeness()` 和调用链传入 `submission_profile`。
  - 当 profile 为 `ei_conference` 时：
    - 不再要求独立 `related work`
    - 不再要求独立 `limitations`
    - 认可 `Results and Analysis`
    - 不输出 “Missing required sections for NeurIPS/ICLR” 这类误导日志
  - 保留真正的结构性检查，例如无标题、正文过短、章节被截断。

- 修 Stage 23/24 的 citation-bib 同步。
  - 修改 `researchclaw/pipeline/stage_impls/_review_publish.py` 的 `_remove_citations_from_text()`，支持多 key markdown citation，如 `[a, b, c]` 和带空格变体。
  - Stage 23 在写 `paper_final_verified.md` 时，正文和 `references_verified.bib` 必须基于同一批保留 key。
  - Stage 24 在复制 `references_verified.bib` 后，按最终 `paper_repaired.md` 实际引用 key 再过滤一次 `references.bib`，确保 `.md/.tex/.bib` 三者一致。
  - 不再允许 Stage 24 成品仍引用不存在于 Stage 24 bibliography 的 key。

- 提前做 BibTeX 去重，并给重复 key 明确优先级。
  - 提取一个共享的 bibliography helper，供 Stage 22、Stage 23、Stage 24 复用。
  - 规则固定为：同 key 重复时优先保留带 DOI/正式期刊版本，丢弃 arXiv 预印本版本。
  - Stage 24 编译前必须先对 `references.bib` 去重；deliverables 阶段只保留兜底检查，不再作为首个 dedup 点。

- 修 DOCX 页数检测与 Stage 24 失败语义。
  - 修改 `researchclaw/pipeline/stage_impls/_final_editorial_repair.py` 的页数检测逻辑，优先使用 `pdfinfo` 读取 `paper_repaired.pdf` 页数；仅在 `pdfinfo` 不可用时才回退现有字节计数。
  - 当 DOCX/PDF 已成功导出但页数无法判定时，记录 warning，不要直接把 Stage 24 判死；只有明确超页或导出失败才 hard-fail。
  - 页数检测恢复后，再让现有 `docx_page_limit` 压缩循环真正生效。

## Test Plan

- 单元测试：`_enforce_reference_limit()` 处理含 `#`、`##`、空行、表格、图片、citation cluster 的 markdown 后，章节结构不丢失。
- 单元测试：`check_paper_completeness(..., submission_profile=\"ei_conference\")` 对 `Introduction / Method / Experiments / Results and Analysis / Conclusion` 结构不再报缺 `related work` 和 `limitations`。
- 单元测试：`_remove_citations_from_text()` 能从 `[a, conrad2022benchmarking, b]` 中只删目标 key，并清理多余逗号/空格。
- 单元测试：BibTeX 去重时对重复的 `aguilarruiz2024classspecific` 保留 DOI 版本，丢弃 arXiv 版本。
- 单元测试：页数检测对 LibreOffice 导出的 PDF 优先走 `pdfinfo`，能读出正确页数。
- 集成测试：构造一个最小 Stage 22/23/24 fixture，验证 Stage 24 最终产物满足：
  - `paper_repaired.md` 保持多章节 markdown
  - `paper_repaired.tex` 编译无 orphan cite
  - `references.bib` 无重复 key
  - `docx_quality.json` 不再出现 `docx_page_count_unavailable`

## Assumptions

- 这次修复以当前 AGENTS 里的定制主线为准，不回退 `ei_conference` 结构，也不把问题归咎于“NeurIPS 模板天然不兼容 EI”。
- `target_conference=neurips_2025` 继续只管模板族，内容结构由 `submission_profile=ei_conference` 决定。
- 当前日志块不涉及数据集下载链路；数据集不是这次修复的目标。
- 计划落地后，建议至少从 `Stage 22` 之后重跑；若要验证 Stage 23 的 citation 同步，一起从 `Stage 23` 重跑更稳。
