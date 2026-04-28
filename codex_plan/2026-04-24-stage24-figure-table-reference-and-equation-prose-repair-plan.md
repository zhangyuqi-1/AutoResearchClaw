# Stage 24 图表显式引用与公式前后 Prose 修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Stage 24 中 canonical markdown 丢失 `Figure N` / `Table N` 正文显式引用、图注块被吞掉、以及展示公式前后 prose 机械化的问题，保证 DOCX 和 PDF 继续同源且内容不再退化。

**Architecture:** 修复全部落在 `paper_repaired.md` 的规范化、恢复和审计链上，不做 DOCX-only 或 PDF-only 的内容补丁。Stage 24 先从 prior source 恢复缺失的 figure/table caption 与正文引用，再用新的审计规则阻止“只有 the figure below / the table below”这类伪讨论通过，最后把公式相邻 prose 从机械模板改成自然 prose，同时保留公式编号这一导出能力。

**Tech Stack:** Python 3.12, pytest, ResearchClaw Stage 24 pipeline, Pandoc, TeX Live

---

### Task 1: 先把当前退化行为锁成失败用例

**Files:**
- Modify: `tests/test_rc_executor.py`
- Inspect: `artifacts/rc-20260415-085845-0de33f/stage-23/paper_final_verified.md`
- Inspect: `artifacts/rc-20260415-085845-0de33f/stage-24/paper_repaired.md`

- [ ] **Step 1: 添加“图只有 generic discussion、不算被正文引用”的失败测试**

```python
def test_stage24_audit_flags_figure_without_explicit_numbered_reference() -> None:
    markdown = (
        "# T\n\n"
        "## Results and Analysis\n\n"
        "The heatmap below reinforces this regime-specific view.\n\n"
        "![Dataset-level F1 heatmap for Adult and Covtype across the evaluated methods.]"
        "(charts/fig_dataset_method_heatmap.png)\n\n"
        "*Figure 4. Dataset-level F1 heatmap for the evaluated methods.*\n"
    )

    issues = stage24_mod._audit_markdown(stage24_mod._split_blocks(markdown))

    assert any(
        issue["type"] == "missing_explicit_figure_reference"
        and issue["severity"] == "high"
        for issue in issues
    )
```

- [ ] **Step 2: 添加“表只有 caption、没有 Table N 正文引用”的失败测试**

```python
def test_stage24_audit_flags_table_without_explicit_numbered_reference() -> None:
    markdown = (
        "# T\n\n"
        "## Experiments\n\n"
        "The benchmark setup is summarized below.\n\n"
        "Table 1. Evaluated conditions in the validated benchmark execution.\n\n"
        "| Abbr. | Condition |\n"
        "|---|---|\n"
        "| CATB | baseline |\n"
    )

    issues = stage24_mod._audit_markdown(stage24_mod._split_blocks(markdown))

    assert any(
        issue["type"] == "missing_explicit_table_reference"
        and issue["severity"] == "high"
        for issue in issues
    )
```

- [ ] **Step 3: 添加“从 prior source 恢复 figure/table 引用和图注块”的失败测试**

```python
def test_stage24_normalize_final_markdown_restores_missing_figure_caption_and_reference() -> None:
    current_markdown = (
        "# T\n\n"
        "## Results and Analysis\n\n"
        "The ablation plot below shows the selector ordering.\n\n"
        "![Selector ablation comparing embedded top-k, PRSM, PSIR, SHDW, and MBDS variants.]"
        "(charts/fig_ablation_selector_variants.png)\n"
    )
    prior_markdown = (
        "# T\n\n"
        "## Results and Analysis\n\n"
        "As shown in Figure 5, adding robustness-oriented or disagreement-oriented structure "
        "does not automatically improve predictive F1.\n\n"
        "![Selector ablation comparing embedded top-k, PRSM, PSIR, SHDW, and MBDS variants.]"
        "(charts/fig_ablation_selector_variants.png)\n\n"
        "Figure 5. Selector-family comparison across embedded top-k ranking, PRISM stability-"
        "filtered correlation pruning, prior-shift ranking, shadow-twin redundancy adjudication, "
        "and minority-boundary disagreement selection.\n"
    )

    normalized = stage24_mod._normalize_final_paper_markdown(
        current_markdown,
        topic="Explainable feature selection for imbalanced tabular classification",
        domains=("ml",),
        submission_profile="default",
        table_caption_sources=(prior_markdown,),
        figure_reference_sources=(prior_markdown,),
    )

    assert "As shown in Figure 5" in normalized
    assert "Figure 5. Selector-family comparison" in normalized
```

- [ ] **Step 4: 添加“公式前后 prose 去机械化”的失败测试**

```python
def test_stage24_normalize_final_markdown_rewrites_equation_prose_to_natural_style() -> None:
    markdown = (
        "# T\n\n"
        "## Method\n\n"
        "The ensemble prediction is computed as follows in Equation (8):\n\n"
        "$$\n"
        "p(y=1 \\mid x) = \\sum_{m=1}^{M} w_m p_m(y=1 \\mid x_{S_m})\n"
        "$$\n\n"
        "In Equation (8), $M$ is the number of compact feature views and $w_m$ is its ensemble weight.\n"
    )

    normalized = stage24_mod._normalize_final_paper_markdown(
        markdown,
        topic="Explainable feature selection for imbalanced tabular classification",
        domains=("ml",),
        submission_profile="default",
    )

    assert "The ensemble prediction is computed as follows:" in normalized
    assert "Equation (8)" not in normalized
    assert "Here, $M$ is the number of compact feature views" in normalized
```

- [ ] **Step 5: 跑测试确认当前实现确实失败**

Run:

```bash
pytest -q tests/test_rc_executor.py -k "explicit_numbered_reference or restores_missing_figure_caption_and_reference or rewrites_equation_prose_to_natural_style"
```

Expected:
- FAIL，且失败点落在 Stage 24 audit / normalize 逻辑，而不是测试写错

---

### Task 2: 恢复 figure/table caption 与正文显式引用，替换掉 generic-only 讨论

**Files:**
- Modify: `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
- Test: `tests/test_rc_executor.py`

- [ ] **Step 1: 给 figure 建 prior-source 恢复映射**

实现约束：
- 新增 `_build_figure_caption_map(sources: tuple[str, ...]) -> dict[str, str]`
- 新增 `_build_figure_reference_map(sources: tuple[str, ...]) -> dict[str, str]`
- `caption_map` 以图片 basename 为 key，例如 `fig_ablation_selector_variants.png`
- `reference_map` 存同一图片附近、包含 `Figure N` 的正文句，优先取图片前 2 个 block 内的句子
- 只接受显式编号引用；`the figure below`、`the heatmap below` 不入 map

- [ ] **Step 2: 给 table 建正文显式引用恢复映射**

实现约束：
- 新增 `_build_table_reference_map(sources: tuple[str, ...]) -> dict[str, str]`
- 仍用现有 `_table_block_signature()` 作为 key
- value 只接受包含 `Table N` 的正文句
- 只恢复正文引用，不覆盖现有 `_restore_table_captions_in_body()` 的 caption 恢复职责

- [ ] **Step 3: 在 canonical normalize 阶段补回缺失的 figure caption block 和正文引用**

实现顺序固定如下：
1. 先做公式规范化
2. 再做 table caption 恢复
3. 再做 missing figure caption 恢复
4. 再做 figure/table 显式引用恢复
5. 最后再做 keywords 去重和 profile 收口

决策细节：
- 新增 `figure_reference_sources: tuple[str, ...] = ()` 参数给 `_normalize_final_paper_markdown()`
- 从 `run_dir / stage-24|23|22` 读取 source，和 table caption source 顺序一致
- 若当前正文已含 `Figure N` / `Table N`，不重复插入
- 若 prior source 有可信原句，优先原样恢复
- 若 prior source 没有可信原句，则插入最短 fallback：
  - 图：`Figure N summarizes <caption_text_without_prefix>.`
  - 表：`Table N reports <caption_text_without_prefix>.`
- fallback 只能用已有 caption / alt text 生成，不允许发明新实验结论

- [ ] **Step 4: 改审计逻辑，把“只有 generic explanation”判成高严重度失败**

实现约束：
- 新增 `_find_first_explicit_figure_reference_index()` 与 `_find_first_explicit_table_reference_index()`
- `Figure N` / `Table N` 是唯一合格的显式引用信号
- `_audit_markdown()` 新增两类 issue：
  - `missing_explicit_figure_reference`
  - `missing_explicit_table_reference`
- severity 固定为 `high`
- `far_from_first_reference` 保留，但只在显式引用存在时才参与距离审计

- [ ] **Step 5: 调整“缺 explanation 自动补句子”的行为**

实现约束：
- `_build_explanation()` 继续可用，但其生成句子不再被当作“显式正文引用”的替代品
- `The figure below summarizes ...` 这类句子只能算 local explanation，不能让 audit 过关
- `_add_missing_explanations()` 保留，用于版面可读性，不负责满足显式引用约束

- [ ] **Step 6: 跑测试确认恢复链和新审计通过**

Run:

```bash
pytest -q tests/test_rc_executor.py -k "missing_explicit_figure_reference or missing_explicit_table_reference or restores_missing_figure_caption_and_reference"
```

Expected:
- PASS

---

### Task 3: 把展示公式前后 prose 从机械模板改成自然 prose

**Files:**
- Modify: `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
- Test: `tests/test_rc_executor.py`

- [ ] **Step 1: 重写紧邻 display equation 的 lead-in 规范化**

实现约束：
- `_normalize_equation_lead_in()` 不再强行拼接 `in Equation (n):`
- 如果原句是：
  - `... in Equation (8):`
  - `... in Equation (10).`
  - `... The loss is written as follows in Equation (10).`
  统一去掉 `in Equation (n)`，保留自然句尾 `:` 或 `.`
- 该改写只作用于与当前 equation block 紧邻的 lead-in，不动跨段远距离交叉引用

- [ ] **Step 2: 重写紧邻 display equation 的 explanation 规范化**

实现约束：
- `_normalize_equation_explanation_block()` 把
  - `In Equation (8), ...`
  - `In this equation, ...`
  - `where ...`
  统一规范成自然 prose
- 规范输出优先：
  - `Here, ...`
  - 或 `In this expression, ...`
- 明确过滤所有伪解释：
  - `denotes a variable defined in the surrounding text`
  - `quad denotes the probe function`
  - `ge denotes ...`
  - `arg / max / in / mathcal / star / tilde / operatorname` 这类符号名

- [ ] **Step 3: 保留已有有效解释，不再自动发明整段参数说明**

实现约束：
- 当原 explanation 已经语义完整时，仅做轻量规范化，不重写
- 当 explanation 缺失时，只允许补最短自然 prose，且只描述明确能从上下文识别的变量
- 禁止再生成 “一个方程里把 6-7 个 token 都解释一遍” 的模板段

- [ ] **Step 4: 调整 Codex editorial prompt，停止重新引入坏风格**

在 `_build_codex_editorial_prompt()` 中固定改成：
- 保留“每个展示公式需要 local lead-in 和必要说明”
- 删除“必须显式写 Equation (n)”这类强约束
- 新增“避免机械化的 Equation (n) 模板句式”
- 新增“每个 retained figure/table 必须在 nearby body text 中被显式写作 Figure N / Table N”

- [ ] **Step 5: 跑测试确认公式 prose 归一化通过**

Run:

```bash
pytest -q tests/test_rc_executor.py -k "rewrites_equation_prose_to_natural_style or normalize_final_markdown_repairs_display_equation_flow or normalize_final_markdown_references_numbered_equations"
```

Expected:
- PASS
- 新测试通过
- 旧的 equation tests 若与新口径冲突，需要同步改断言为自然 prose 口径

---

### Task 4: 全量回归并用真实 artifact 复核

**Files:**
- Verify: `researchclaw/pipeline/stage_impls/_final_editorial_repair.py`
- Verify: `tests/test_rc_executor.py`
- Verify: `artifacts/rc-20260415-085845-0de33f`

- [ ] **Step 1: 跑 Stage 24 相关单测回归**

Run:

```bash
pytest -q tests/test_rc_executor.py tests/test_rc_templates.py
```

Expected:
- PASS

- [ ] **Step 2: 验证同一个 artifact 时，先避免续修旧坏稿**

Run:

```bash
mv artifacts/rc-20260415-085845-0de33f/stage-24 artifacts/rc-20260415-085845-0de33f/stage-24.before-reference-prose-repair
```

Expected:
- 原坏 `stage-24` 被保留备份，新的 rerun 将从 `stage-23` / `stage-22` 重新进入 24 步，而不是续修旧坏稿

- [ ] **Step 3: 重跑真实 Stage 24**

Run:

```bash
set -a && source .env && set +a && researchclaw run --config config2.arc.yaml --resume --from-stage FINAL_EDITORIAL_REPAIR --auto-approve
```

Expected:
- Stage 24 重新生成 `paper_repaired.md/.docx/.tex/.pdf`

- [ ] **Step 4: 人工核对 rerun 结果的 6 个验收点**

检查以下文件：
- `artifacts/rc-20260415-085845-0de33f/stage-24/paper_repaired.md`
- `artifacts/rc-20260415-085845-0de33f/stage-24/paper_repaired.docx`
- `artifacts/rc-20260415-085845-0de33f/stage-24/paper_repaired.pdf`
- `artifacts/rc-20260415-085845-0de33f/stage-24/paper_repaired_docx.md`

验收点：
- `Figure 3/4/5` 在正文里重新出现显式引用，不再只有 `below`
- 方法部分的流程图和主图重新有 `Figure 1/2` 的正文引用
- `Table 1/2/3` 仍有 caption，且正文也有 `Table N` 引用
- `The ensemble prediction is computed as follows:` 这类自然 prose 恢复，不再是 `in Equation (8)`
- `In Equation (n), ...` 伪模板消失，改成 `Here, ...` / `In this expression, ...`
- DOCX 和 PDF 继续使用同一份 canonical 内容，没有内容分叉

- [ ] **Step 5: 若 rerun 仍失败，按固定顺序排查**

排查顺序固定：
1. 先看 `paper_repaired.md` 是否已恢复 figure/table 引用
2. 再看 `paper_repaired_docx.md` 是否忠实继承同样内容
3. 再看 `paper_repaired.tex` 是否保留相同引用和自然公式 prose
4. 最后才看 DOCX/PDF 排版后处理

---

### Task 5: 收尾说明与文档同步

**Files:**
- Modify: `AGENTS.md`
- Verify: `codex_plan/2026-04-24-stage24-figure-table-reference-and-equation-prose-repair-plan.md`

- [ ] **Step 1: 在实现完成后补记一条 Stage 24 新口径**

写入 `AGENTS.md` 的内容要点：
- Stage 24 不允许用 generic figure/table explanation 替代显式正文引用
- Stage 24 会从 prior source 恢复缺失的 figure caption 与 `Figure N` / `Table N` 正文句
- 公式编号仍保留，但 canonical prose 默认走自然表达，不再强制 `Equation (n)` 模板句

- [ ] **Step 2: 最终确认这份计划与实现一致**

Run:

```bash
rg -n "missing_explicit_figure_reference|missing_explicit_table_reference|figure_reference_sources|avoid mechanical Equation" researchclaw/pipeline/stage_impls/_final_editorial_repair.py tests/test_rc_executor.py AGENTS.md
```

Expected:
- 能看到计划中的关键机制都已经真实落地，而不是只停留在计划文档

---

## Self-Review

- 需求覆盖：
  - 已覆盖图引用丢失、表引用丢失、figure caption block 丢失、公式前后 prose 机械化、24 步续修旧稿导致旧坏内容被继承。
- 占位符扫描：
  - 无 `TODO` / `TBD` / “后续再看”。
- 一致性检查：
  - 新增审计 issue 名称固定为 `missing_explicit_figure_reference` 和 `missing_explicit_table_reference`。
  - 新增 normalize 参数名固定为 `figure_reference_sources`。
  - 验证顺序固定为“canonical markdown -> docx markdown -> tex -> 排版后处理”。
