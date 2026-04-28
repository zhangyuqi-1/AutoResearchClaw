# Stage 24 Round-2 Repair Plan

## Summary

修复当前 `EXPORT_PUBLISH -> CITATION_VERIFY -> FINAL_EDITORIAL_REPAIR` 链路中的 4 个真实问题：

1. `Stage 24` 接收了结构损坏的 markdown，但缺少足够严格的 integrity guard。
2. 图文 explanation 审计误报，不能正确识别紧邻图片的 `Figure N` 说明和图片 alt 文本。
3. completeness 的章节词数检查没有把 `###` 子节内容并入主节，导致 `Method / Experiments / Results and Analysis` 被误判为 0 词。
4. Stage 24 的页数约束结果没有被明确归因和稳定收口，导致最终仍可能因为 `docx_page_limit_exceeded` 失败。

## Key Changes

- 为 `Stage 24` 新增 markdown integrity 校验。
  - 校验代码 fence 是否成对。
  - 校验标题、`Abstract`、`Introduction` 与主章节结构仍可解析。
  - 对 Codex 每轮输出先做 integrity gate；若损坏则丢弃该轮输出，保留上一轮 markdown。

- 修 `missing_explanation` 审计逻辑。
  - `_extract_bundles()` 记录图片 alt 文本。
  - `_is_explanation_for_bundle()` 同时使用：
    - 图片 alt 文本关键词
    - 独立 caption block
    - 图片前后紧邻段落中的 `Figure N` 提及
  - 无独立 caption 时，仍允许通过紧邻说明段落判定图已被解释。

- 修 completeness 章节词数统计。
  - `check_paper_completeness()` 对 `Method`、`Experiments`、`Results and Analysis` 这类主节聚合其后续 H3/H4 子节正文。
  - 继续保留 EI 模式下的结构豁免，不回退 `submission_profile: ei_conference`。

- 收紧 Stage 24 成功条件与失败归因。
  - 结构损坏不再进入 `paper_repaired.md`。
  - 高危图文问题只保留真实未解决项，不再接受当前误报。
  - `docx_page_limit_exceeded` 继续作为真实失败条件保留，并在 assessment 中明确体现。

## Test Plan

- 新增 Stage 24 测试：
  - Codex 输出未闭合代码 fence 时，应拒收该轮 markdown。
  - 图片无 caption block、但前后紧邻段落明确讨论 `Figure N` 时，应判为已解释。

- 新增 completeness 测试：
  - 主节正文为空、但多个子节有足够内容时，不应再报 `0 words`。

- 回归执行：
  - Stage 24/export 相关定向 pytest。
  - converter completeness 相关定向 pytest。

## Assumptions

- 本轮不处理数据集、Stage 10、OpenCode 与代理问题。
- 真实重跑仍建议从 `EXPORT_PUBLISH` 开始，避免续用坏的旧 `stage-24` 产物。
