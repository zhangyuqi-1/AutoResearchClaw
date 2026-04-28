# 本地 Sandbox 数据准备与失败收紧实施计划

> **面向后续执行者：** 这份文档沉淀的是“实验代码生成与执行链”上的一条独立改造线，不是论文导出链。它解决的问题是：本地 `sandbox` 过去只会直接跑 `main.py`，导致生成代码无法稳定准备真实数据，进一步诱发“小数据集替代回退”“setup 缺失却继续跑”“Stage 10/12 误判成功”等问题。

**目标：** 让本地 `sandbox` 能真正支持 `requirements.txt -> setup.py -> main.py` 的三阶段项目执行；同时禁止生成代码在目标数据集失败后偷偷换成别的数据集，并把这类问题在 Stage 10、Stage 12、experiment_diagnosis 三处前置拦截。

**架构原则：**
1. 数据准备属于 `setup.py`，不属于 `main.py`。
2. `main.py` 默认应视为离线消费阶段，只读取已准备好的数据和缓存。
3. 目标数据集不可用时，应直接失败，而不是换成另一个无关数据集继续跑。
4. 本地 `sandbox` 应成为真实可执行环境，而不是只做“把代码抄进去然后碰运气跑一下”的弱校验器。

**技术范围：**
- `researchclaw/config.py`
- `config.researchclaw.example.yaml`
- `researchclaw/experiment/sandbox.py`
- `researchclaw/prompts.py`
- `researchclaw/experiment/validator.py`
- `researchclaw/pipeline/stage_impls/_code_generation.py`
- `researchclaw/pipeline/stage_impls/_execution.py`
- `researchclaw/pipeline/experiment_diagnosis.py`
- `researchclaw/pipeline/experiment_repair.py`
- 对应回归测试

---

## 一、问题背景

本轮问题来自一次真实运行：

- run id: `rc-20260415-085845-0de33f`
- 典型症状：
  - Stage 9 明确指出可下载数据集没有在 `setup.py` 中准备。
  - 生成代码里尝试加载目标数据集失败后，退回到 `load_breast_cancer()` / `load_wine()`。
  - Stage 10 只做逐文件语法检查，没有抓住项目级问题。
  - Stage 12 可能在退出码为 `0` 时仍把这种“回退到错误数据集”的运行当成成功。
  - diagnosis 侧没有把“数据集替代”和“缺 setup phase”明确提升成技术报告级问题。

根因不是单点，而是 4 个环节同时偏松：

1. 本地 `sandbox` 过去只跑 `main.py`，不跑 `requirements.txt` 和 `setup.py`。
2. prompt 仍在暗示 `/workspace/data` 这种 Docker 口径，而不是本地可复用的数据根目录。
3. 代码生成后的校验只看单文件，不看项目级契约。
4. 诊断分类对“数据集替代回退”没有单独收口。

---

## 二、设计结论

### 1. 本地 sandbox 必须升级为三阶段执行

本地 `sandbox` 不再只是：

1. 复制文件
2. 执行 `python main.py`

而必须升级为：

1. 若存在 `requirements.txt`，先执行依赖安装阶段
2. 若存在 `setup.py`，执行数据准备阶段
3. 最后执行 `main.py`

### 2. 数据准备统一走共享环境变量

后续所有生成代码和 repair prompt 都应统一使用：

- `RC_DATA_DIR`
- `HF_HOME`
- `HF_CACHE`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `SKLEARN_HOME`
- `PIP_CACHE_DIR`

禁止继续硬编码：

- `'/workspace/data'`
- 任意写死的用户目录或绝对缓存目录

### 3. 不允许数据集替代回退

以下行为明确禁止：

1. Adult 加载失败后切换到 `breast_cancer`
2. Covtype 加载失败后切换到 `wine`
3. 任意 `using ... fallback for ...` 形式的数据集替代

正确行为只能是：

1. 在 `setup.py` 中准备目标数据集
2. 在 `main.py` 中读取目标数据集
3. 若目标数据集仍不可用，抛出 `RuntimeError`

### 4. 三个阶段都要收紧

必须同时收紧：

1. Stage 10：生成后项目级验证
2. Stage 12：执行结果状态判定
3. `experiment_diagnosis`：失败归因与论文模式降级

不能只改其中一个，否则仍会漏判。

---

## 三、实施拆分

### 任务 1：扩展 sandbox 配置项

**责任：** 给本地 sandbox 三阶段执行和数据目录注入提供正式配置入口。

**涉及文件：**

1. `researchclaw/config.py`
2. `config.researchclaw.example.yaml`

**要做的事：**

1. 在 `SandboxConfig` 中新增：
   - `network_policy`
   - `auto_install_deps`
   - `pip_timeout_sec`
   - `setup_timeout_sec`
   - `data_root`
2. 在 `_parse_experiment_config()` 中接通这些字段。
3. 在示例配置中补齐对应键，避免后续会话误以为不存在。

**落地结果：**

已完成。当前默认语义为：

- `network_policy: "full"`
- `auto_install_deps: true`
- `pip_timeout_sec: 300`
- `setup_timeout_sec: 300`
- `data_root: ""`

---

### 任务 2：把本地 sandbox 升级为三阶段执行器

**责任：** 让本地 `ExperimentSandbox.run_project()` 真正支持项目级执行。

**涉及文件：**

1. `researchclaw/experiment/sandbox.py`

**要做的事：**

1. 在 `run_project()` 中增加三阶段逻辑：
   - phase 0: `python -m pip install -r requirements.txt`
   - phase 1: `python -u setup.py`
   - phase 2: `python -u main.py`
2. 注入统一环境变量：
   - `RC_DATA_DIR`
   - HF / transformers / sklearn cache 目录
   - `PIP_CACHE_DIR`
3. 为各阶段分别落日志文件，便于排查。
4. 对本地 sandbox 的 `network_policy` 限定为：
   - `full`
   - `none`
5. 若 `network_policy='none'` 且项目包含 `requirements.txt` 或 `setup.py`，直接失败。

**落地结果：**

已完成。当前本地 sandbox 已不再只做单阶段 `main.py` 直跑。

---

### 任务 3：统一 prompt 里的数据准备口径

**责任：** 不再让模型继续生成 Docker 语义的路径或偷偷替代数据集的代码。

**涉及文件：**

1. `researchclaw/prompts.py`

**要做的事：**

1. 把数据目录指引从 `/workspace/data` 改成 `os.environ['RC_DATA_DIR']`。
2. 强化 prompt 约束：
   - 禁止数据集替代回退
   - 失败时抛 `RuntimeError`
   - 可下载数据集必须在 `setup.py` 中准备
3. 要求生成显式日志信号，例如：
   - `DATASET_READY`
   - `DATASET_LOAD`

**落地结果：**

已完成。当前 prompt 已明确站在“setup 负责准备，main 负责消费”的口径上。

---

### 任务 4：新增项目级代码契约校验

**责任：** 把“逐文件语法没问题，但项目整体注定跑不通”的问题前置拦截。

**涉及文件：**

1. `researchclaw/experiment/validator.py`
2. `researchclaw/pipeline/stage_impls/_code_generation.py`

**要做的事：**

1. 在 `validator.py` 中新增 `validate_project_files(files)`。
2. 让它检查以下问题：
   - root-level 相对导入
   - 有可下载数据集但没有 `setup.py`
   - `download_if_missing=False` 但仍无 setup phase
   - 硬编码 `'/workspace/data'`
   - `load_breast_cancer()` / `load_wine()` 等替代回退模式
3. 在 Stage 10 中调用该项目级验证。
4. 对无 `setup.py` / `requirements.txt` 的 flat project 增加 smoke run。
5. 若项目级校验失败或 smoke run 失败，Stage 10 直接失败。

**落地结果：**

已完成。Stage 10 现在不再只依赖逐文件 Python 校验。

---

### 任务 5：收紧 Stage 12 的成功 / 失败判定

**责任：** 防止“退出码是 0，但其实已经换错数据集”的运行被误判成功。

**涉及文件：**

1. `researchclaw/pipeline/stage_impls/_execution.py`

**要做的事：**

1. 扩展执行输出扫描。
2. 若 stdout / stderr 中出现以下信号，强制 `run_status = "failed"`：
   - `DATA_WARNING: ... fallback`
   - `using ... fallback for`
   - `load_breast_cancer`
   - `load_wine`

**落地结果：**

已完成。Stage 12 当前已把“数据集替代回退”作为硬失败处理。

---

### 任务 6：在诊断层显式建模两类新缺陷

**责任：** 让 diagnosis 和论文模式选择真正理解这类问题。

**涉及文件：**

1. `researchclaw/pipeline/experiment_diagnosis.py`
2. `researchclaw/pipeline/experiment_repair.py`

**要做的事：**

1. 新增两个 deficiency type：
   - `DATASET_SUBSTITUTION`
   - `SETUP_PHASE_MISSING`
2. 把它们接入 `_select_paper_mode()`：
   - 一旦出现，论文模式直接压到 `TECHNICAL_REPORT`
3. 收紧 `_check_synthetic_data()`，避免把“数据集替代”和“纯 synthetic fallback”混为一类。
4. 在 repair prompt 里加上预期数据集与正确缓存路径约束。
5. 同时让 `assess_experiment_quality()` 能从 `experiment_summary` 顶层 `stdout/stderr` 读到这些信号，不只依赖 `best_run`。

**落地结果：**

已完成。diagnosis 现在已经把这两类问题单独分类，并会触发技术报告降级。

---

## 四、关键文件边界

### 1. `researchclaw/experiment/sandbox.py`

这里是“本地 sandbox 是否真的支持真实数据准备”的核心文件。

不要再把它理解成纯粹的“安全执行器”。在当前分支里，它已经承担：

1. 项目复制
2. harness 注入
3. 数据目录环境注入
4. phase 执行
5. phase 日志写出

### 2. `researchclaw/prompts.py`

这里不是装饰性文案层，而是影响 Stage 10 代码生成质量的第一道约束源。

如果这里继续保留 `/workspace/data` 或对 fallback 表述模糊，后面再怎么补 validator 都会反复吃亏。

### 3. `researchclaw/experiment/validator.py`

这里负责“生成后、执行前”的硬门禁。

它应该阻止：

1. 契约不成立的项目
2. 路径口径错误的项目
3. 明显作弊式回退的项目

而不是只做语法校验。

### 4. `researchclaw/pipeline/stage_impls/_code_generation.py`

这里的角色是把 validator 的项目级结论真正变成 Stage 10 的失败条件。

否则 validator 只是写报告，不起真正 gate 作用。

### 5. `researchclaw/pipeline/stage_impls/_execution.py`

这里负责“运行完成后是否算成功”的最终口径。

不能只看退出码。

### 6. `researchclaw/pipeline/experiment_diagnosis.py`

这里负责把执行失败翻译成结构化原因，并影响论文模式选择。

如果不在这里建模，后续 Stage 15 仍可能把错误实验误写成正常论文。

---

## 五、验证方案

### 1. 配置与解析测试

**文件：**

- `tests/test_rc_config.py`

**覆盖点：**

1. `SandboxConfig` 新字段默认值
2. `RCConfig.from_dict()` 对新字段的解析

### 2. 本地 sandbox 执行链测试

**文件：**

- `tests/test_entry_point_validation.py`

**覆盖点：**

1. `setup.py` 先于 `main.py` 执行
2. `RC_DATA_DIR` 在 setup/main 间共享
3. phase log 正常写出
4. `network_policy='none'` 时遇到 `setup.py` / `requirements.txt` 直接失败

### 3. 项目级 validator 测试

**文件：**

- `tests/test_rc_validator.py`

**覆盖点：**

1. root-level 相对导入报错
2. 可下载数据集但无 `setup.py` 报错
3. `fetch_covtype(download_if_missing=False)` 但无 `setup.py` 报错
4. 硬编码 `'/workspace/data'` 报错
5. `load_breast_cancer` / `load_wine` 替代回退报错
6. 合法 `setup.py + RC_DATA_DIR` 组合通过

### 4. diagnosis 测试

**文件：**

- `tests/test_experiment_diagnosis.py`

**覆盖点：**

1. `DATA_WARNING: using wine fallback ...` -> `DATASET_SUBSTITUTION`
2. `does not include setup.py` -> `SETUP_PHASE_MISSING`
3. `DATASET_SUBSTITUTION` 会把论文模式压到 `TECHNICAL_REPORT`

### 5. 模块语法验证

**命令：**

```bash
python -m py_compile \
  researchclaw/experiment/sandbox.py \
  researchclaw/experiment/validator.py \
  researchclaw/pipeline/stage_impls/_code_generation.py \
  researchclaw/pipeline/stage_impls/_execution.py \
  researchclaw/pipeline/experiment_diagnosis.py \
  researchclaw/pipeline/experiment_repair.py
```

---

## 六、已完成状态

本计划当前已经实现并验证。

### 已落地代码

1. `researchclaw/config.py`
2. `config.researchclaw.example.yaml`
3. `researchclaw/experiment/sandbox.py`
4. `researchclaw/prompts.py`
5. `researchclaw/experiment/validator.py`
6. `researchclaw/pipeline/stage_impls/_code_generation.py`
7. `researchclaw/pipeline/stage_impls/_execution.py`
8. `researchclaw/pipeline/experiment_diagnosis.py`
9. `researchclaw/pipeline/experiment_repair.py`

### 已落地测试

1. `tests/test_rc_config.py`
2. `tests/test_entry_point_validation.py`
3. `tests/test_rc_validator.py`
4. `tests/test_experiment_diagnosis.py`

### 已执行验证

```bash
pytest tests/test_rc_config.py tests/test_entry_point_validation.py tests/test_rc_validator.py -q
pytest tests/test_experiment_diagnosis.py -q
python -m py_compile \
  researchclaw/experiment/sandbox.py \
  researchclaw/experiment/validator.py \
  researchclaw/pipeline/stage_impls/_code_generation.py \
  researchclaw/pipeline/stage_impls/_execution.py \
  researchclaw/pipeline/experiment_diagnosis.py \
  researchclaw/pipeline/experiment_repair.py
```

验证结果：

1. 配置 / sandbox / validator 相关定向测试通过
2. diagnosis 定向测试通过
3. 关键模块 `py_compile` 通过

---

## 七、后续使用建议

1. 若要验证这条链，建议至少从 Stage 10 重跑，不建议直接从 Stage 14 或 Stage 24 开始。
2. 若运行里再次出现小数据集替代，不要先怀疑 pipeline 不支持下载；先看生成代码是否违反了新约束。
3. 若本地 sandbox 被显式配置成 `network_policy: none`，就不要再期待它在本地阶段联网准备新数据。
4. 若 diagnosis 再次把实验压成 `TECHNICAL_REPORT`，优先检查是否仍有：
   - `DATASET_SUBSTITUTION`
   - `SETUP_PHASE_MISSING`

---

## 八、一句话总结

这条改造线的核心不是“给生成代码更多容错”，而是反过来收紧整个实验链：

- 本地 sandbox 负责真实准备数据
- Stage 10 负责前置拦截坏项目
- Stage 12 负责拒绝伪成功运行
- diagnosis 负责把这类问题明确降级成技术报告

只有这样，后续论文链拿到的实验结果才有基本可信度。
