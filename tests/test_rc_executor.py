# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import executor as rc_executor
from researchclaw.pipeline.stage_impls import _final_editorial_repair as stage24_mod
from researchclaw.pipeline.stages import Stage, StageStatus


class FakeLLMClient:
    def __init__(self, response_text: str = "mock response"):
        self.response_text: str = response_text
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object):
        _ = kwargs
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.response_text, model="fake-model")


class FakeLLMClientWithConfig(FakeLLMClient):
    def __init__(self, response_text: str = "mock response"):
        super().__init__(response_text=response_text)
        self.config: SimpleNamespace = SimpleNamespace(
            base_url="http://fake", api_key="fake-key"
        )


@dataclass(frozen=True)
class FakeEditorialLoopResult:
    success: bool
    markdown: str
    review: dict[str, object]
    iterations: list[dict[str, object]]
    assessment: dict[str, object]
    error: str = ""


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {
            "topic": "test-driven science",
            "paper_title": "Exact Configured Test Title",
            "domains": ["ml", "systems"],
            "daily_paper_count": 2,
            "quality_threshold": 8.2,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {
            "channel": "local",
            "on_stage_start": True,
            "on_stage_fail": False,
            "on_gate_required": True,
        },
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": True, "use_message": True},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _write_prior_artifact(
    run_dir: Path, stage_num: int, filename: str, content: str
) -> None:
    stage_dir = run_dir / f"stage-{stage_num:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / filename).write_text(content, encoding="utf-8")


def test_executor_map_has_24_entries() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert len(executor_map) == 24


def test_every_stage_member_has_matching_executor() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert set(executor_map.keys()) == set(Stage)


def test_stage_result_dataclass_fields() -> None:
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
    )
    assert result.stage == Stage.TOPIC_INIT
    assert result.status == StageStatus.DONE
    assert result.artifacts == ("goal.md",)
    assert result.error is None
    assert result.decision == "proceed"
    assert result.evidence_refs == ()


def test_utcnow_iso_returns_valid_iso_timestamp() -> None:
    ts = rc_executor._utcnow_iso()
    assert ts.endswith("+00:00")
    assert "T" in ts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("before\n```yaml\na: 1\n```\nafter", "a: 1"),
        ("```yml\nkey: value\n```", "key: value"),
        ("```\nplain: true\n```", "plain: true"),
        ("  x: y  ", "x: y"),
    ],
)
def test_extract_yaml_block_variants(text: str, expected: str) -> None:
    assert rc_executor._extract_yaml_block(text) == expected


@pytest.mark.parametrize(
    ("payload", "default", "expected"),
    [
        ('{"ok": true}', {"fallback": True}, {"ok": True}),
        ("[1, 2, 3]", {"fallback": True}, [1, 2, 3]),
        ("not-json", {"fallback": True}, {"fallback": True}),
    ],
)
def test_safe_json_loads_valid_and_invalid(payload: str, default, expected) -> None:
    assert rc_executor._safe_json_loads(payload, default) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("../secret", "__secret"),
        ("name with spaces!.md", "name_with_spaces_.md"),
        ("", "unnamed"),
    ],
)
def test_safe_filename_sanitization(raw: str, expected: str) -> None:
    assert rc_executor._safe_filename(raw) == expected


def test_safe_filename_truncates_to_100_chars() -> None:
    raw = "x" * 120
    cleaned = rc_executor._safe_filename(raw)
    assert len(cleaned) == 100
    assert cleaned == "x" * 100


def test_build_context_preamble_basic_fields(
    rc_config: RCConfig, run_dir: Path
) -> None:
    text = rc_executor._build_context_preamble(rc_config, run_dir)
    assert "## Research Context" in text
    assert "test-driven science" in text
    assert "ml, systems" in text


def test_build_context_preamble_includes_selected_prior_artifacts(
    rc_config: RCConfig, run_dir: Path
) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "goal content")
    _write_prior_artifact(run_dir, 8, "hypotheses.md", "hyp content")
    _write_prior_artifact(run_dir, 7, "synthesis.md", "synth content")
    text = rc_executor._build_context_preamble(
        rc_config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_synthesis=True,
    )
    assert "### Goal" in text
    assert "goal content" in text
    assert "### Hypotheses" in text
    assert "hyp content" in text
    assert "### Synthesis" in text
    assert "synth content" in text


def test_read_prior_artifact_finds_newest_file(run_dir: Path) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "old")
    _write_prior_artifact(run_dir, 3, "goal.md", "new")
    found = rc_executor._read_prior_artifact(run_dir, "goal.md")
    assert found == "new"


def test_read_prior_artifact_finds_directory_path(run_dir: Path) -> None:
    cards_dir = run_dir / "stage-06" / "cards"
    cards_dir.mkdir(parents=True)
    (cards_dir / "card-1.json").write_text("{}", encoding="utf-8")
    found = rc_executor._read_prior_artifact(run_dir, "cards/")
    assert found == str(cards_dir)


def test_read_prior_artifact_returns_none_when_not_found(run_dir: Path) -> None:
    assert rc_executor._read_prior_artifact(run_dir, "missing.md") is None


def test_read_best_analysis_prefers_best_file(run_dir: Path) -> None:
    """BUG-225: _read_best_analysis prefers analysis_best.md at run root."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    # Create degenerate analysis in stage-14 and best at run root
    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Degenerate analysis", encoding="utf-8")
    (run_dir / "analysis_best.md").write_text("Best analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Best analysis"


def test_read_best_analysis_falls_back_to_prior_artifact(run_dir: Path) -> None:
    """BUG-225: Falls back to _read_prior_artifact when no analysis_best.md."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Only analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Only analysis"


def test_read_best_analysis_returns_empty_when_none(run_dir: Path) -> None:
    """BUG-225: Returns empty string when no analysis exists at all."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    result = _read_best_analysis(run_dir)
    assert result == ""


def test_write_stage_meta_writes_expected_json(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-01"
    stage_dir.mkdir()
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md",),
        decision="proceed",
        evidence_refs=("stage-01/goal.md",),
    )
    rc_executor._write_stage_meta(stage_dir, Stage.TOPIC_INIT, "run-abc", result)
    payload = cast(
        dict[str, Any],
        json.loads((stage_dir / "decision.json").read_text(encoding="utf-8")),
    )
    assert payload["stage_id"] == "01-topic_init"
    assert payload["run_id"] == "run-abc"
    assert payload["status"] == "done"
    assert payload["decision"] == "proceed"
    assert payload["output_artifacts"] == ["goal.md"]
    assert payload["evidence_refs"] == ["stage-01/goal.md"]
    assert payload["next_stage"] == 2
    assert re.match(r"\d{4}-\d{2}-\d{2}T", payload["ts"])


def test_write_stage_meta_keeps_paused_stage_as_next_stage(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-02"
    stage_dir.mkdir()
    result = rc_executor.StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.PAUSED,
        artifacts=("refinement_log.json",),
        decision="resume",
        error="ACP prompt timed out after 1800s",
        evidence_refs=("stage-02/refinement_log.json",),
    )
    rc_executor._write_stage_meta(
        stage_dir, Stage.PROBLEM_DECOMPOSE, "run-paused", result
    )
    payload = cast(
        dict[str, Any],
        json.loads((stage_dir / "decision.json").read_text(encoding="utf-8")),
    )
    assert payload["status"] == "paused"
    assert payload["decision"] == "resume"
    assert payload["next_stage"] == int(Stage.PROBLEM_DECOMPOSE)


def test_execute_stage_creates_stage_dir_writes_artifacts_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fake_llm = FakeLLMClientWithConfig("# Goal\n\nMocked goal body")
    monkeypatch.setattr(
        "researchclaw.pipeline.executor.LLMClient.from_rc_config",
        lambda _config: fake_llm,
    )

    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-1",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert "goal.md" in result.artifacts
    assert "hardware_profile.json" in result.artifacts
    assert (run_dir / "stage-01").is_dir()
    assert (
        (run_dir / "stage-01" / "goal.md")
        .read_text(encoding="utf-8")
        .startswith("# Goal")
    )
    assert (run_dir / "stage-01" / "hardware_profile.json").exists()
    assert len(fake_llm.calls) == 1

    decision = cast(
        dict[str, Any],
        json.loads(
            (run_dir / "stage-01" / "decision.json").read_text(encoding="utf-8")
        ),
    )
    assert decision["run_id"] == "run-1"
    assert decision["status"] == "done"
    assert decision["output_artifacts"] == ["goal.md", "hardware_profile.json"]


def test_execute_stage_contract_validation_missing_output_file_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
        )

    monkeypatch.setitem(rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, bad_executor)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-2",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing or empty output: goal.md" in (result.error or "")


def test_execute_stage_contract_validation_missing_output_directory_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 5, "shortlist.jsonl", '{"title": "x"}')

    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.DONE,
            artifacts=("cards/",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.KNOWLEDGE_EXTRACT, bad_executor
    )
    result = rc_executor.execute_stage(
        Stage.KNOWLEDGE_EXTRACT,
        run_dir=run_dir,
        run_id="run-3",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing output directory: cards/" in (result.error or "")


def test_execute_stage_missing_required_input_returns_failed(
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    result = rc_executor.execute_stage(
        Stage.PROBLEM_DECOMPOSE,
        run_dir=run_dir,
        run_id="run-4",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing input: goal.md" in (result.error or "")


def test_execute_stage_gate_behavior_auto_approve_true_keeps_done(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-5",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    memory_entries = getattr(adapters.memory, "entries", [])
    assert any(
        ns == "gates" and "auto-approved" in content for ns, content in memory_entries
    )


def test_execute_stage_gate_behavior_auto_approve_false_blocks(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-6",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=False,
    )
    assert result.status == StageStatus.BLOCKED_APPROVAL
    assert result.decision == "block"
    message_calls = getattr(adapters.message, "calls", [])
    assert message_calls
    assert "Approval required" in message_calls[-1][2]


def test_execute_stage_llm_client_creation_error_falls_back_without_crash(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def boom(_config: RCConfig):
        raise RuntimeError("llm init failed")

    monkeypatch.setattr("researchclaw.pipeline.executor.LLMClient.from_rc_config", boom)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-7",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    assert (run_dir / "stage-01" / "goal.md").exists()


def test_execute_stage_executor_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def raising_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        raise RuntimeError("stage exploded")

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, raising_executor
    )
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-8",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert result.decision == "retry"
    assert "stage exploded" in (result.error or "")


@pytest.mark.parametrize(
    "stage",
    [
        Stage.TOPIC_INIT,
        Stage.PROBLEM_DECOMPOSE,
        Stage.SEARCH_STRATEGY,
        Stage.LITERATURE_COLLECT,
        Stage.LITERATURE_SCREEN,
        Stage.KNOWLEDGE_EXTRACT,
        Stage.SYNTHESIS,
        Stage.HYPOTHESIS_GEN,
        Stage.EXPERIMENT_DESIGN,
        Stage.CODE_GENERATION,
    ],
)
def test_stage_executor_mapping_values_are_callable(stage: Stage) -> None:
    assert callable(rc_executor._STAGE_EXECUTORS[stage])


class TestStageHealth:
    def test_stage_health_json_written(self, tmp_path: Path) -> None:
        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )
        result = execute_stage(
            Stage.TOPIC_INIT,
            run_dir=tmp_path,
            run_id="test-health",
            config=config,
            adapters=AdapterBundle(),
            auto_approve_gates=True,
        )
        health_path = tmp_path / "stage-01" / "stage_health.json"
        assert result is not None
        assert health_path.exists()

    def test_stage_health_has_required_fields(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "research_questions": ["q1"]}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-health-fields",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert "stage_id" in data
            assert "run_id" in data
            assert "duration_sec" in data
            assert "status" in data
            assert "timestamp" in data
            assert data["duration_sec"] >= 0


    def test_stage_health_duration_positive(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "sub_problems": []}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-duration",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert data["duration_sec"] >= 0

# Contracts import for Stage 13/22 preservation features.
from researchclaw.pipeline.contracts import CONTRACTS


class TestIterativeRefine:
    def _prepare_refine_inputs(self, run_dir: Path) -> None:
        _write_prior_artifact(
            run_dir,
            10,
            "experiment.py",
            (
                "import random\n"
                "random.seed(42)\n"
                "for i in range(5):\n"
                "    print(f'val_loss: {0.5 - i*0.05:.4f}')\n"
            ),
        )
        (run_dir / "stage-12" / "runs").mkdir(parents=True, exist_ok=True)
        _write_prior_artifact(
            run_dir,
            12,
            "runs/run-1.json",
            json.dumps(
                {
                    "run_id": "run-1",
                    "status": "completed",
                    "metrics": {"val_loss": 0.35},
                }
            ),
        )

    def test_refine_simulated_mode_skips(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        """R10-Fix3: Simulated mode should skip iterative refinement entirely."""
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Force simulated mode to test the skip behavior
        import copy
        sim_cfg = copy.deepcopy(rc_config)
        object.__setattr__(sim_cfg.experiment, "mode", "simulated")

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sim_cfg,
            adapters,
            llm=None,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["skipped"] is True
        assert payload["mode"] == "simulated"
        assert result.status == StageStatus.DONE
        # Original code should be copied as final
        assert (stage_dir / "experiment_final.py").exists()

    def test_refine_no_llm_saves_original_as_final(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=None,
        )

        original_code = (run_dir / "stage-10" / "experiment.py").read_text(
            encoding="utf-8"
        )
        final_code = (stage_dir / "experiment_final.py").read_text(encoding="utf-8")
        assert original_code == final_code
        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["stop_reason"] == "llm_unavailable"
        assert result.status == StageStatus.DONE

    def test_refine_acp_timeout_pauses_for_resume(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        from researchclaw.pipeline.stage_impls import _execution as execution_impl

        def _timeout(*args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("ACP prompt timed out after 1800s")

        monkeypatch.setattr(execution_impl, "_chat_with_prompt", _timeout)

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=FakeLLMClient("unused"),
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert result.status == StageStatus.PAUSED
        assert result.decision == "resume"
        assert result.artifacts == ("refinement_log.json",)
        assert payload["paused"] is True
        assert payload["stop_reason"] == "acp_prompt_timeout"
        assert payload["pause_iteration"] == 1
        assert payload["best_version"] == "experiment/"
        assert not (stage_dir / "experiment_final").exists()

    def test_refine_with_llm_generates_improved_code(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        assert (stage_dir / "experiment_v1").is_dir()
        assert (stage_dir / "experiment_final.py").exists()
        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert isinstance(payload.get("iterations"), list)
        assert payload["iterations"]

    def test_refine_converges_after_no_improvement(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        import sys

        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        sandbox_data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test-driven science",
                "domains": ["ml", "systems"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 3,
                "metric_key": "val_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        sandbox_config = RCConfig.from_dict(
            sandbox_data,
            project_root=tmp_path,
            check_paths=False,
        )
        llm = FakeLLMClient(
            "```python\nfor _ in range(3):\n    print('val_loss: 0.5000')\n```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sandbox_config,
            adapters,
            llm=llm,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["converged"] is True
        assert payload["stop_reason"] == "no_improvement_for_2_iterations"

    def test_refine_artifacts_include_version_files(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=llm,
        )

        assert "refinement_log.json" in result.artifacts
        assert "experiment_final/" in result.artifacts
        assert any(
            artifact.startswith("experiment_v") and artifact.endswith("/")
            for artifact in result.artifacts
        )

    def test_refine_sandbox_mode_runs_code(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        import sys

        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        sandbox_data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test-driven science",
                "domains": ["ml", "systems"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 3,
                "metric_key": "val_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        sandbox_config = RCConfig.from_dict(
            sandbox_data,
            project_root=tmp_path,
            check_paths=False,
        )
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sandbox_config,
            adapters,
            llm=llm,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert any(
            "sandbox" in iteration for iteration in payload.get("iterations", [])
        )


class TestExportPublishCodePackage:
    @staticmethod
    def _stub_export_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
        from researchclaw.experiment import visualize as rc_visualize
        from researchclaw.pipeline.stage_impls import _review_publish as review_publish
        from researchclaw.templates import compiler as rc_compiler
        import researchclaw.templates as rc_templates

        class _FakeTemplate:
            name = "neurips_2025"
            display_name = "NeurIPS 2025"

            @staticmethod
            def get_style_files() -> list[Path]:
                return []

        def _fake_markdown_to_latex(
            text: str,
            tpl: object,
            title: str = "",
            authors: str = "",
            bib_file: str = "",
            bib_entries: dict[str, str] | None = None,
        ) -> str:
            _ = tpl, title, authors, bib_file, bib_entries
            lines = [r"\documentclass{article}", r"\usepackage{graphicx}", r"\begin{document}"]
            for match in re.finditer(r"!\[[^\]]*\]\((charts/[^)]+)\)", text):
                lines.append(r"\begin{figure}")
                lines.append(rf"\includegraphics{{{match.group(1)}}}")
                lines.append(r"\end{figure}")
            lines.append(r"\end{document}")
            return "\n".join(lines)

        def _fake_compile_latex(tex_path: Path, max_attempts: int = 2) -> SimpleNamespace:
            _ = max_attempts
            (tex_path.parent / "paper.pdf").write_bytes(b"%PDF-1.4\n")
            return SimpleNamespace(success=True, errors=[])

        def _fake_quality(_: Path) -> SimpleNamespace:
            return SimpleNamespace(
                page_count=1,
                unresolved_refs=[],
                unresolved_cites=[],
                overfull_hboxes=[],
                orphan_figures=[],
                orphan_labels=[],
                warnings_summary=[],
            )

        monkeypatch.setattr(rc_templates, "get_template", lambda _: _FakeTemplate())
        monkeypatch.setattr(rc_templates, "markdown_to_latex", _fake_markdown_to_latex)
        monkeypatch.setattr(rc_compiler, "compile_latex", _fake_compile_latex)
        monkeypatch.setattr(rc_compiler, "check_compiled_quality", _fake_quality)
        monkeypatch.setattr(rc_visualize, "generate_all_charts", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            review_publish,
            "_generate_framework_diagram_prompt",
            lambda *args, **kwargs: "# Framework Diagram Prompt\n",
        )

    def test_export_packages_experiment_final(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            'import numpy\nprint("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert (stage_dir / "code" / "experiment.py").exists()
        assert (stage_dir / "code" / "README.md").exists()
        req_text = (stage_dir / "code" / "requirements.txt").read_text(encoding="utf-8")
        assert "numpy" in req_text

    def test_export_falls_back_to_experiment_py(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            10,
            "experiment.py",
            'import numpy\nprint("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        code_text = (stage_dir / "code" / "experiment.py").read_text(encoding="utf-8")
        assert "val_loss: 0.1" in code_text

    def test_export_no_experiment_skips_code_dir(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = rc_executor._execute_export_publish(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=None,
        )

        assert not (stage_dir / "code").exists()
        assert "code/" not in result.artifacts

    def test_export_detects_multiple_dependencies(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            (
                "import numpy\n"
                "import torch\n"
                "from sklearn.metrics import accuracy_score\n"
                "print(accuracy_score([1], [1]))\n"
            ),
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        requirements = (stage_dir / "code" / "requirements.txt").read_text(
            encoding="utf-8"
        )
        assert "numpy" in requirements
        assert "torch" in requirements
        assert "scikit-learn" in requirements

    def test_export_code_readme_contains_title(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# My Great Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            'print("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        readme = (stage_dir / "code" / "README.md").read_text(encoding="utf-8")
        assert "My Great Paper" in readme

    def test_export_copies_referenced_non_fig_stage14_images(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "![Pipeline](charts/pipeline_overview_2.png)\n\n"
                "![Loss](charts/fig_val_loss_single_run.png)\n"
            ),
        )
        stage14 = run_dir / "stage-14_v2" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "pipeline_overview_2.png").write_bytes(b"pipeline")
        (stage14 / "fig_val_loss_single_run.png").write_bytes(b"loss")

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert (stage_dir / "charts" / "pipeline_overview_2.png").exists()
        assert (stage_dir / "charts" / "fig_val_loss_single_run.png").exists()
        tex = (stage_dir / "paper.tex").read_text(encoding="utf-8")
        assert "charts/pipeline_overview_2.png" in tex
        assert "charts/fig_val_loss_single_run.png" in tex

    def test_export_strips_orphan_heading_and_framework_placeholder(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "#\n\n"
                "![Figure 9: Concept Illustration 1](charts/concept_illustration_1.png)\n\n"
                "## Related Work\n\n"
                "Text.\n\n"
                "![Framework Overview](charts/framework_diagram.png)\n\n"
                "**Figure N.** Overview of the proposed methodology.\n"
            ),
        )
        stage14 = run_dir / "stage-14_v2" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "concept_illustration_1.png").write_bytes(b"concept")

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert "\n#\n" not in paper_final
        assert "framework_diagram.png" not in paper_final
        assert "Figure N." not in paper_final

    def test_export_strips_meta_sections_and_skips_checklist_append(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment import visualize as rc_visualize
        from researchclaw.pipeline.stage_impls import _review_publish as review_publish
        from researchclaw.templates import compiler as rc_compiler
        import researchclaw.templates as rc_templates

        captured_markdown: dict[str, str] = {}

        class _FakeTemplate:
            name = "neurips_2025"
            display_name = "NeurIPS 2025"

            @staticmethod
            def get_style_files() -> list[Path]:
                return []

        def _fake_markdown_to_latex(
            text: str,
            tpl: object,
            title: str = "",
            authors: str = "",
            bib_file: str = "",
            bib_entries: dict[str, str] | None = None,
        ) -> str:
            _ = tpl, title, authors, bib_file, bib_entries
            captured_markdown["text"] = text
            return "\n".join(
                [
                    r"\documentclass{article}",
                    r"\begin{document}",
                    text,
                    r"\end{document}",
                ]
            )

        def _fake_compile_latex(tex_path: Path, max_attempts: int = 2) -> SimpleNamespace:
            _ = max_attempts
            (tex_path.parent / "paper.pdf").write_bytes(b"%PDF-1.4\n")
            return SimpleNamespace(success=True, errors=[])

        def _fake_quality(_: Path) -> SimpleNamespace:
            return SimpleNamespace(
                page_count=1,
                unresolved_refs=[],
                unresolved_cites=[],
                overfull_hboxes=[],
                orphan_figures=[],
                orphan_labels=[],
                warnings_summary=[],
            )

        monkeypatch.setattr(rc_templates, "get_template", lambda _: _FakeTemplate())
        monkeypatch.setattr(rc_templates, "markdown_to_latex", _fake_markdown_to_latex)
        monkeypatch.setattr(rc_compiler, "compile_latex", _fake_compile_latex)
        monkeypatch.setattr(rc_compiler, "check_compiled_quality", _fake_quality)
        monkeypatch.setattr(rc_visualize, "generate_all_charts", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            review_publish,
            "_generate_framework_diagram_prompt",
            lambda *args, **kwargs: "# Framework Diagram Prompt\n",
        )

        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "Actual paper body.\n\n"
                "## Lessons from Prior Runs\n\n"
                "Leakage.\n\n"
                "## Learned Skills from Prior Runs\n\n"
                "More leakage.\n\n"
                "## NeurIPS Paper Checklist\n\n"
                "Answer: [Yes]\n"
            ),
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert "Lessons from Prior Runs" not in paper_final
        assert "Learned Skills from Prior Runs" not in paper_final
        assert "NeurIPS Paper Checklist" not in paper_final
        assert "Actual paper body." in paper_final
        assert "NeurIPS Paper Checklist" not in captured_markdown["text"]

    def test_export_uses_stage14_manifest_caption_for_referenced_figures(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "Actual paper body.\n\n"
                "![fig_hyp2_physics_ablation](charts/fig_hyp2_physics_ablation.png)\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "fig_hyp2_physics_ablation.png").write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "figure_id": "fig_hyp2_physics_ablation",
                        "file_path": "charts/fig_hyp2_physics_ablation.png",
                        "caption": (
                            "Ablation focused on Hypothesis 2 comparing "
                            "reliability gating without physics, reliability "
                            "gating with weak physics, and full optical "
                            "residual PINN modeling."
                        ),
                        "title": "Weak Physics vs Full Optical Modeling",
                        "paper_section": "Analysis",
                    }
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert "Ablation focused on Hypothesis 2 comparing" in paper_final

    def test_export_does_not_inject_unanchored_stage14_figures(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            "# Test Paper\n\n## Results\n\nOnly text discussion.\n",
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "fig_main_results.png").write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "figure_id": "fig_main_results",
                        "file_path": "charts/fig_main_results.png",
                        "caption": "Main results comparison across methods.",
                    }
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert "charts/fig_main_results.png" not in paper_final
        assert "Main results comparison across methods." not in paper_final

    def test_export_cleans_prompt_style_caption_suffixes_for_referenced_figures(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Method\n\n"
                "Actual paper body.\n\n"
                "![comparison_illustration_5](charts/comparison_illustration_5.png)\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "comparison_illustration_5.png").write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "figure_id": "comparison_illustration_5",
                        "file_path": "charts/comparison_illustration_5.png",
                        "caption": (
                            "Side-by-side comparison illustration of weak "
                            "physics-informed regularization versus full "
                            "optical residual PINN-style modeling, showing "
                            "the relative complexity, assumptions, and data "
                            "flow. The figure should help readers understand "
                            "the engineering tradeoff between lightweight "
                            "feed-forward constraints and heavier explicit "
                            "optical modeling."
                        ),
                        "paper_section": "Method",
                    }
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert (
            "Side-by-side comparison illustration of weak physics-informed "
            "regularization versus full optical residual PINN-style modeling, "
            "showing the relative complexity, assumptions, and data flow."
        ) in paper_final

    def test_export_backfills_to_minimum_four_figures_with_local_explanations(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Method\n\n"
                "PACT combines semantic and structural evidence.\n\n"
                "**Figure 1.** Overview of PACT.\n\n"
                "## Setup\n\n"
                "Each regime holds out one project and evaluates transfer.\n\n"
                "## Metrics\n\n"
                "PR-AUC and ROC-AUC are reported alongside F1.\n\n"
                "## Results\n\n"
                "As shown in Figure 3, PACT leads on F1 while Figure 4 highlights leakage.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "**Figure 3.** Main F1 comparison.\n\n"
                "![Leakage](charts/fig_semantic_vs_fusion_scatter.png)\n\n"
                "**Figure 4.** Leakage remains high.\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        for name in (
            "architecture_diagram_1.png",
            "pipeline_overview_2.png",
            "fig_main_f1_results.png",
            "fig_semantic_vs_fusion_scatter.png",
            "fig_multi_metric_grouped_results.png",
        ):
            (stage14 / name).write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "file_path": "charts/architecture_diagram_1.png",
                        "caption": "PACT combines semantic and structural views through an agreement-aware gate.",
                    },
                    {
                        "file_path": "charts/pipeline_overview_2.png",
                        "caption": "End-to-end experimental pipeline for held-out cross-project evaluation.",
                    },
                    {
                        "file_path": "charts/fig_main_f1_results.png",
                        "caption": "Main F1 comparison across methods.",
                    },
                    {
                        "file_path": "charts/fig_semantic_vs_fusion_scatter.png",
                        "caption": "Leakage remains high even when predictive performance is strong.",
                    },
                    {
                        "file_path": "charts/fig_multi_metric_grouped_results.png",
                        "caption": "Grouped comparison across F1, PR-AUC, MCC, and ROC-AUC.",
                    },
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        assert paper_final.count("![](") + paper_final.count("![") >= 4
        assert "The model architecture is summarized visually below." in paper_final
        assert "The end-to-end evaluation protocol is summarized visually below." in paper_final
        assert "charts/architecture_diagram_1.png" in paper_final
        assert "charts/pipeline_overview_2.png" in paper_final
        assert "**Figure 1.** Overview of PACT." in paper_final
        assert (
            paper_final.index("The model architecture is summarized visually below.")
            < paper_final.index("charts/architecture_diagram_1.png")
            < paper_final.index("**Figure 1.** Overview of PACT.")
        )

    def test_export_moves_architecture_figure_next_to_first_method_mention(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Method\n\n"
                "PACT combines semantic and structural evidence.\n\n"
                "**Figure 1.** Overview of PACT. Functions are encoded separately before gated fusion.\n\n"
                "### Training Procedure\n\n"
                "Optimization details.\n\n"
                "![Late architecture](charts/architecture_diagram_1.png)\n\n"
                "**Figure 2.** Late architecture placement.\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        (stage14 / "architecture_diagram_1.png").write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "file_path": "charts/architecture_diagram_1.png",
                        "caption": "PACT combines semantic and structural views through an agreement-aware gate.",
                    }
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        explanation_pos = paper_final.index("The model architecture is summarized visually below.")
        image_pos = paper_final.index("charts/architecture_diagram_1.png")
        method_pos = paper_final.index("**Figure 1.** Overview of PACT.")
        training_pos = paper_final.index("### Training Procedure")
        assert explanation_pos < image_pos < method_pos
        assert image_pos < training_pos
        assert paper_final.count("charts/architecture_diagram_1.png") == 1
        assert "**Figure 2.** Late architecture placement." not in paper_final
        assert "The figure should help readers understand" not in paper_final

    def test_export_moves_pipeline_figure_to_setup_not_introduction(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_export_pipeline(monkeypatch)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Introduction\n\n"
                "Cross-project transfer is the main challenge.\n\n"
                "## Setup\n\n"
                "Each held-out regime evaluates transfer under a fixed protocol.\n\n"
                "## Results\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "**Figure 1.** Main F1 comparison.\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        for name in ("pipeline_overview_2.png", "fig_main_f1_results.png", "architecture_diagram_1.png"):
            (stage14 / name).write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "file_path": "charts/pipeline_overview_2.png",
                        "caption": "End-to-end experimental pipeline for held-out cross-project evaluation.",
                    },
                    {
                        "file_path": "charts/fig_main_f1_results.png",
                        "caption": "Main F1 comparison across methods.",
                    },
                    {
                        "file_path": "charts/architecture_diagram_1.png",
                        "caption": "PACT combines semantic and structural views through an agreement-aware gate.",
                    },
                ]
            ),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        intro_section = paper_final.split("## Introduction", 1)[1].split("## Setup", 1)[0]
        setup_section = paper_final.split("## Setup", 1)[1].split("## Results", 1)[0]
        assert "charts/pipeline_overview_2.png" not in intro_section
        assert "charts/pipeline_overview_2.png" in setup_section
        assert "The end-to-end evaluation protocol is summarized visually below." in setup_section
        assert (
            setup_section.index("The end-to-end evaluation protocol is summarized visually below.")
            < setup_section.index("charts/pipeline_overview_2.png")
        )

    def test_export_does_not_backfill_abstract_with_images_or_break_math_tex(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import researchclaw.templates as rc_templates
        from researchclaw.templates import compiler as rc_compiler
        from researchclaw.templates import get_template, markdown_to_latex

        self._stub_export_pipeline(monkeypatch)
        monkeypatch.setattr(rc_templates, "get_template", get_template)
        monkeypatch.setattr(rc_templates, "markdown_to_latex", markdown_to_latex)
        _write_prior_artifact(
            run_dir,
            19,
            "paper_revised.md",
            (
                "# Test Paper\n\n"
                "## Abstract\n\n"
                "Abstract text.\n\n"
                "## Method\n\n"
                "Let $x_i=(t_i,g_i)$ and $W_s h_i^{(s)} + b_s$ define the model.\n\n"
                "**Figure 1.** Overview of PACT.\n\n"
                "## Setup\n\n"
                "Each held-out regime evaluates transfer.\n\n"
                "## Results\n\n"
                "As shown in Figure 3, the main comparison is competitive.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "**Figure 3.** Main F1 comparison.\n\n"
                "![Leakage](charts/fig_semantic_vs_fusion_scatter.png)\n\n"
                "**Figure 4.** Leakage remains high.\n"
            ),
        )

        stage14 = run_dir / "stage-14" / "charts"
        stage14.mkdir(parents=True, exist_ok=True)
        for name in (
            "architecture_diagram_1.png",
            "pipeline_overview_2.png",
            "fig_main_f1_results.png",
            "fig_semantic_vs_fusion_scatter.png",
        ):
            (stage14 / name).write_bytes(b"chart")
        (stage14 / "figure_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "file_path": "charts/architecture_diagram_1.png",
                        "caption": "PACT combines semantic and structural views through an agreement-aware gate.",
                    },
                    {
                        "file_path": "charts/pipeline_overview_2.png",
                        "caption": "End-to-end experimental pipeline for held-out cross-project evaluation.",
                    },
                    {
                        "file_path": "charts/fig_main_f1_results.png",
                        "caption": "Main F1 comparison across methods.",
                    },
                    {
                        "file_path": "charts/fig_semantic_vs_fusion_scatter.png",
                        "caption": "Leakage remains high even when predictive performance is strong.",
                    },
                ]
            ),
            encoding="utf-8",
        )

        real_compile = rc_compiler.compile_latex

        def _compile_with_one_attempt(tex_path: Path, max_attempts: int = 2) -> SimpleNamespace:
            _ = max_attempts
            result = real_compile(tex_path, max_attempts=1, timeout=30)
            return SimpleNamespace(success=result.success, errors=result.errors)

        monkeypatch.setattr(rc_compiler, "compile_latex", _compile_with_one_attempt)

        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        paper_final = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
        abstract_section = paper_final.split("## Abstract", 1)[1].split("## Method", 1)[0]
        assert "charts/pipeline_overview_2.png" not in abstract_section

        tex = (stage_dir / "paper.tex").read_text(encoding="utf-8")
        assert "![" not in tex
        assert r"$x_i=(t_i,g_i)$" in tex
        assert r"$W_s h_i^{(s)} + b_s$" in tex
        assert r"$x\_i" not in tex
        assert "Missing \\endcsname inserted." not in tex


def test_contracts_stage13_includes_experiment_final() -> None:
    assert "experiment_final/" in CONTRACTS[Stage.ITERATIVE_REFINE].output_files


def test_contracts_stage22_includes_code_dir() -> None:
    assert "code/" in CONTRACTS[Stage.EXPORT_PUBLISH].output_files


class TestStage24EditorialRepair:
    def _stub_editorial_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import researchclaw.templates as rc_templates
        from researchclaw.templates import compiler as rc_compiler

        class _FakeTemplate:
            name = "neurips_2025"
            display_name = "NeurIPS 2025"

            @staticmethod
            def get_style_files() -> list[Path]:
                return []

        def _fake_markdown_to_latex(
            text: str,
            tpl: object,
            title: str = "",
            authors: str = "",
            bib_file: str = "",
            bib_entries: dict[str, str] | None = None,
        ) -> str:
            _ = tpl, title, authors, bib_file, bib_entries
            return "\n".join(
                [
                    r"\documentclass{article}",
                    r"\begin{document}",
                    text,
                    r"\end{document}",
                ]
            )

        def _fake_compile_latex(
            tex_path: Path,
            max_attempts: int = 2,
            timeout: int = 120,
        ) -> SimpleNamespace:
            _ = max_attempts, timeout
            (tex_path.parent / "paper_repaired.pdf").write_bytes(b"%PDF-1.4\n")
            return SimpleNamespace(success=True, errors=[])

        monkeypatch.setattr(rc_templates, "get_template", lambda _: _FakeTemplate())
        monkeypatch.setattr(rc_templates, "markdown_to_latex", _fake_markdown_to_latex)
        monkeypatch.setattr(rc_compiler, "compile_latex", _fake_compile_latex)

    def test_stage24_uses_codex_editorial_loop_and_writes_outputs(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text(
            (
                "# Test Paper\n\n"
                "## Introduction\n\n"
                "Cross-project transfer is difficult.\n\n"
                "![Pipeline](charts/pipeline_overview_2.png)\n\n"
                "**Figure 1.** Evaluation pipeline.\n\n"
                "## Setup\n\n"
                "Each held-out regime evaluates transfer under a fixed protocol.\n\n"
                "## Results\n\n"
                "PACT remains competitive on the primary metric.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "**Figure 2.** Main F1 comparison across methods.\n"
            ),
            encoding="utf-8",
        )
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        charts = stage22 / "charts"
        charts.mkdir()
        (charts / "pipeline_overview_2.png").write_bytes(b"pipeline")
        (charts / "fig_main_f1_results.png").write_bytes(b"mainf1")

        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text(
            (stage22 / "paper_final.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (stage23 / "verification_report.json").write_text(
            json.dumps({"summary": {"verified": 10, "total": 10}}),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        called: dict[str, object] = {}

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            called["stage_dir"] = stage_dir_arg
            called["run_dir"] = run_dir_arg
            called["source_markdown"] = source_markdown
            called["config"] = config_arg
            repaired = source_markdown.replace(
                "## Introduction\n\n"
                "Cross-project transfer is difficult.\n\n"
                "![Pipeline](charts/pipeline_overview_2.png)\n\n"
                "**Figure 1.** Evaluation pipeline.\n\n"
                "## Setup\n\n"
                "Each held-out regime evaluates transfer under a fixed protocol.\n\n",
                "## Introduction\n\n"
                "Cross-project transfer is difficult.\n\n"
                "## Setup\n\n"
                "Each held-out regime evaluates transfer under a fixed protocol.\n\n"
                "The end-to-end evaluation protocol is summarized visually below.\n\n"
                "![Pipeline](charts/pipeline_overview_2.png)\n\n"
                "**Figure 1.** Evaluation pipeline.\n\n",
            ).replace(
                "PACT remains competitive on the primary metric.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)",
                "PACT remains competitive on the primary metric.\n\n"
                "The figure below compares PACT against the strongest baselines on F1.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)",
            )
            return FakeEditorialLoopResult(
                success=True,
                markdown=repaired,
                review={
                    "source": "stage-23/paper_final_verified.md",
                    "initial_issue_count": 2,
                    "issues": [
                        {"type": "wrong_section_placement", "image": "pipeline_overview_2.png"},
                        {"type": "missing_explanation", "image": "fig_main_f1_results.png"},
                    ],
                },
                iterations=[
                    {"iteration": 1, "action": "codex_editorial_rewrite", "changed": True}
                ],
                assessment={
                    "status": "pass",
                    "remaining_issue_count": 0,
                    "remaining_high_severity_issues": 0,
                    "moved_figures": ["pipeline_overview_2.png"],
                    "rewritten_windows": ["Setup", "Results"],
                    "dropped_figures": [],
                    "compile_clean": True,
                    "improved_vs_stage22": True,
                },
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        repaired = (stage_dir / "paper_repaired.md").read_text(encoding="utf-8")
        intro = repaired.split("## Introduction", 1)[1].split("## Setup", 1)[0]
        setup = repaired.split("## Setup", 1)[1].split("## Results", 1)[0]
        results = repaired.split("## Results", 1)[1]

        assert "charts/pipeline_overview_2.png" not in intro
        assert "charts/pipeline_overview_2.png" in setup
        assert "The end-to-end evaluation protocol is summarized visually below." in setup
        assert "The figure below compares PACT against the strongest baselines on F1." in results
        assert called["stage_dir"] == stage_dir
        assert called["run_dir"] == run_dir
        assert (stage_dir / "paper_editorial_input.md").exists()
        assert (stage_dir / "editorial_review.json").exists()
        assert (stage_dir / "editorial_iterations.json").exists()
        assert (stage_dir / "editorial_final_assessment.json").exists()
        assert (stage_dir / "paper_repaired.tex").exists()
        assert (stage_dir / "paper_repaired.pdf").exists()
        assert result.stage is Stage.FINAL_EDITORIAL_REPAIR
        assert result.status is StageStatus.DONE

    def test_stage24_exports_docx_when_pandoc_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage_dir = tmp_path / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "paper_repaired.md").write_text(
            "# Title\n\nEquation $x_i = y_i$.\n",
            encoding="utf-8",
        )

        def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
            _ = args, kwargs
            (stage_dir / "paper_repaired.docx").write_bytes(b"PK\x03\x04docx")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(stage24_mod, "which", lambda _name: "/usr/bin/pandoc")
        monkeypatch.setattr(stage24_mod.subprocess, "run", _fake_run)

        artifacts, evidence, ok = stage24_mod._export_editorial_docx(stage_dir)

        assert ok is True
        assert artifacts == ["paper_repaired.docx", "docx_quality.json"]
        assert evidence == ["stage-24/paper_repaired.docx", "stage-24/docx_quality.json"]
        assert (stage_dir / "paper_repaired.docx").exists()
        assert (stage_dir / "docx_quality.json").exists()

    def test_stage24_skips_docx_when_pandoc_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage_dir = tmp_path / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "paper_repaired.md").write_text("# Title\n", encoding="utf-8")

        monkeypatch.setattr(stage24_mod, "which", lambda _name: None)

        artifacts, evidence, ok = stage24_mod._export_editorial_docx(stage_dir)

        assert ok is False
        assert artifacts == []
        assert evidence == []

    def test_stage24_prepares_docx_markdown_with_metadata_and_shifted_headings(
        self,
    ) -> None:
        bibliography = (
            "@article{smith2024,\n"
            "  author = {Smith, Jane},\n"
            "  title = {A Test Reference},\n"
            "  journal = {Journal of Tests},\n"
            "  year = {2024}\n"
            "}\n\n"
            "@article{doe2023,\n"
            "  author = {Doe, John},\n"
            "  title = {Another Reference},\n"
            "  journal = {Proceedings of Examples},\n"
            "  year = {2023}\n"
            "}\n"
        )
        markdown = (
            "# Test Title\n\n"
            "## Abstract\n\n"
            "Abstract first paragraph.\n\n"
            "Abstract second paragraph.\n\n"
            "## Introduction\n\n"
            "Intro text with [smith2024, doe2023].\n\n"
            "### Method\n\n"
            "Method text.\n\n"
            "![Overview](charts/fig.png)\n\n"
            "*Figure 1. Overview caption.*\n\n"
            "**Table 1. Setup caption.**\n\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n"
        )

        prepared = stage24_mod._prepare_docx_markdown(
            markdown,
            authors="Anonymous",
            bibliography_name="references.bib",
            bibliography_text=bibliography,
        )

        assert 'title: "Test Title"' in prepared
        assert 'author: "Anonymous"' in prepared
        assert 'custom-style="Abstract"' in prepared
        assert "# Introduction" in prepared
        assert "## Method" in prepared
        assert "*Figure 1. Overview caption.*" not in prepared
        assert "![](charts/fig.png)" in prepared
        assert 'custom-style="ImageCaption"' in prepared
        assert 'custom-style="TableCaption"' in prepared
        assert "<sup>[1, 2]</sup>" in prepared
        assert "# References" in prepared
        assert "[1] Smith, Jane." in prepared
        assert 'custom-style="TableCaption"}\nTable 1 summarizes' not in prepared

    def test_stage24_prepares_docx_markdown_numbers_generic_italic_image_captions(
        self,
    ) -> None:
        markdown = (
            "# Title\n\n"
            "## Results\n\n"
            "Lead-in sentence.\n\n"
            "![Metric comparison](charts/fig.png)\n\n"
            "*Metric comparison across methods.*\n\n"
            "Follow-up sentence.\n"
        )

        prepared = stage24_mod._prepare_docx_markdown(
            markdown,
            authors="Anonymous",
            bibliography_name="references.bib",
        )

        assert "![](charts/fig.png)" in prepared
        assert "*Metric comparison across methods.*" not in prepared
        assert 'custom-style="ImageCaption"}\nFigure 1. Metric comparison across methods' in prepared

    def test_stage24_prepare_docx_markdown_normalizes_bullet_list_items(self) -> None:
        markdown = (
            "# Title\n\n"
            "## Limitations\n\n"
            "- First limitation.\n"
            "- Second limitation.\n"
        )

        prepared = stage24_mod._prepare_docx_markdown(
            markdown,
            authors="Anonymous",
            bibliography_name="references.bib",
        )

        assert "\n- First limitation." not in prepared
        assert "• First limitation." in prepared
        assert "• Second limitation." in prepared

    def test_stage24_docx_export_uses_reference_doc_and_standalone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage_dir = tmp_path / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "paper_repaired.md").write_text("# Title\n", encoding="utf-8")
        (stage_dir / "references.bib").write_text(
            "@article{smith2024, author={Smith, Jane}, title={A Test Reference}, year={2024}}\n",
            encoding="utf-8",
        )
        ref_doc = tmp_path / "reference.docx"
        ref_doc.write_bytes(b"PK\x03\x04ref")
        captured: dict[str, object] = {}

        def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            captured["cmd"] = cmd
            _ = kwargs
            (stage_dir / "paper_repaired.docx").write_bytes(b"PK\x03\x04docx")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(stage24_mod, "which", lambda _name: "/usr/bin/pandoc")
        monkeypatch.setattr(stage24_mod, "_docx_reference_doc_path", lambda: ref_doc)
        monkeypatch.setattr(stage24_mod.subprocess, "run", _fake_run)

        artifacts, evidence, ok = stage24_mod._export_editorial_docx(
            stage_dir,
            authors="Anonymous",
            bibliography_name="references.bib",
        )

        assert ok is True
        assert artifacts == ["paper_repaired.docx", "docx_quality.json"]
        assert evidence == ["stage-24/paper_repaired.docx", "stage-24/docx_quality.json"]
        cmd = cast(list[str], captured["cmd"])
        assert "--standalone" in cmd
        assert "--reference-doc" in cmd
        assert "paper_repaired_docx.md" in cmd

    def test_stage24_prepare_docx_markdown_strips_latex_float_envs(self) -> None:
        markdown = (
            "# Title\n\n"
            "## Results\n\n"
            "Intro paragraph.\n\n"
            "\\begin{figure}[!t]\n"
            "\\centering\n"
            "\\includegraphics[width=0.95\\\\columnwidth]{charts/fig5.png}\n"
            "\\caption{Figure 5. Scatter plot comparing semantic and fusion confidence.}\n"
            "\\label{fig:scatter}\n"
            "\\end{figure}\n\n"
            "## Discussion\n\n"
            "Outro.\n"
        )

        prepared = stage24_mod._prepare_docx_markdown(
            markdown,
            authors="Anonymous",
            bibliography_name="references.bib",
        )

        assert "\\begin{figure}" not in prepared
        assert "\\includegraphics" not in prepared
        assert "charts/fig5.png" in prepared
        assert 'custom-style="ImageCaption"' in prepared
        assert "Figure 5. Scatter plot comparing semantic and fusion confidence." in prepared

    def test_stage24_postprocess_docx_adds_heading_numbering_and_removes_empty_paragraphs(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Paper</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Introduction</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Method</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="BodyText"/></w:pPr><w:r><w:t>Body.</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="BodyText"/></w:pPr></w:p>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/></w:style>
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        quality = stage24_mod._postprocess_editorial_docx(docx_path)

        assert quality["heading_numbering_ok"] is True
        assert quality["clean"] is True
        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
            numbering_xml_after = zf.read("word/numbering.xml").decode("utf-8")
            styles_xml_after = zf.read("word/styles.xml").decode("utf-8")
        assert "<w:numPr>" in doc_xml
        assert "Heading1" in doc_xml and "Heading2" in doc_xml
        assert doc_xml.count("<w:p>") == 4  # empty paragraph removed
        assert 'w:numFmt w:val="decimal"' in numbering_xml_after
        assert "345A8A" not in styles_xml_after
        assert 'w:styleId="BodyText"' in styles_xml_after and 'w:firstLine="360"' in styles_xml_after
        assert 'w:styleId="BodyText"' in styles_xml_after and 'w:sz w:val="24"' in styles_xml_after
        assert 'w:styleId="BodyText"' in styles_xml_after and 'w:line="260"' in styles_xml_after
        assert 'w:styleId="BodyText"' in styles_xml_after and 'w:after="36"' in styles_xml_after
        assert 'w:styleId="Bibliography"' in styles_xml_after and 'w:sz w:val="21"' in styles_xml_after
        assert 'w:styleId="Bibliography"' in styles_xml_after and 'w:line="280"' in styles_xml_after
        assert 'w:styleId="Bibliography"' in styles_xml_after and 'w:after="120"' in styles_xml_after

    def test_stage24_postprocess_docx_formats_tables_like_paper_tables(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="TableCaption"/></w:pPr><w:r><w:t>Table 1. Example.</w:t></w:r></w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:type="auto" w:w="0"/>
      </w:tblPr>
      <w:tblGrid><w:gridCol w:w="2400"/><w:gridCol w:w="2400"/></w:tblGrid>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Method</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>0.9299</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>PACT</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>0.9261</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/></w:style>
  <w:style w:type="paragraph" w:styleId="Compact"><w:name w:val="Compact"/></w:style>
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        quality = stage24_mod._postprocess_editorial_docx(docx_path)

        assert quality["clean"] is True
        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
            styles_xml_after = zf.read("word/styles.xml").decode("utf-8")
        assert "w:tblBorders" in doc_xml
        assert 'w:top' in doc_xml and 'w:bottom' in doc_xml
        assert 'w:tblW w:type="pct" w:w="5000"' in doc_xml
        assert 'w:tblLayout w:type="fixed"' in doc_xml
        assert 'w:insideH w:val="nil"' in doc_xml
        assert 'w:vAlign w:val="center"' in doc_xml
        assert 'w:cantSplit' in doc_xml
        assert doc_xml.count('w:pStyle w:val="Compact"') >= 2
        assert 'w:spacing w:before="0" w:after="0" w:line="200"' in doc_xml
        assert 'w:styleId="Compact"' in styles_xml_after
        assert 'w:w="30" w:type="dxa"' in doc_xml
        assert 'w:sz w:val="16"' in doc_xml
        assert 'w:gridCol w:w="' in doc_xml
        assert 'w:tcW w:type="dxa"' in doc_xml

    def test_stage24_postprocess_docx_widens_long_text_columns_in_tables(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tblPr><w:tblW w:type="auto" w:w="0"/></w:tblPr>
      <w:tblGrid><w:gridCol w:w="2400"/><w:gridCol w:w="2400"/><w:gridCol w:w="2400"/></w:tblGrid>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Method</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>F1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Cross-project F1 variance</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>GraphCodeBERT Semantic Transfer</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>0.9261</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>0.0036</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/></w:style>
  <w:style w:type="paragraph" w:styleId="Compact"><w:name w:val="Compact"/></w:style>
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        stage24_mod._postprocess_editorial_docx(docx_path)

        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")

        widths = [int(m) for m in re.findall(r'w:gridCol w:w="(\d+)"', doc_xml)]
        assert len(widths) >= 3
        assert widths[0] > widths[1]
        assert widths[2] > widths[1]

    def test_stage24_postprocess_docx_sets_page_layout_and_paragraph_keep_rules(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Paper</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Author"/></w:pPr><w:r><w:t>Anonymous</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Introduction</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="FirstParagraph"/></w:pPr><w:r><w:t>Body text.</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="ImageCaption"/></w:pPr><w:r><w:t>Figure 1. Caption.</w:t></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>
  <w:style w:type="paragraph" w:styleId="Author"><w:name w:val="Author"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style>
  <w:style w:type="paragraph" w:styleId="FirstParagraph"><w:name w:val="First Paragraph"/></w:style>
  <w:style w:type="paragraph" w:styleId="ImageCaption"><w:name w:val="Image Caption"/></w:style>
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        quality = stage24_mod._postprocess_editorial_docx(docx_path)

        assert quality["clean"] is True
        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
            styles_xml_after = zf.read("word/styles.xml").decode("utf-8")
        assert "w:pgMar" in doc_xml
        assert "w:pgSz" in doc_xml
        assert 'w:jc w:val="both"' in styles_xml_after
        assert "<w:keepNext/>" in styles_xml_after
        assert "<w:keepLines/>" in styles_xml_after

    def test_stage24_postprocess_docx_superscripts_inline_numeric_citations(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="BodyText"/></w:pPr>
      <w:r><w:t>Semantic code models perform well</w:t></w:r>
      <w:r><w:t xml:space="preserve"> </w:t></w:r>
      <w:r><w:t>[1, 2, 3]</w:t></w:r>
      <w:r><w:t xml:space="preserve"> in prior work.</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr><w:pStyle w:val="Bibliography"/></w:pPr>
      <w:r><w:t>[1] Reference entry.</w:t></w:r>
    </w:p>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        stage24_mod._postprocess_editorial_docx(docx_path)

        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")

        assert 'w:vertAlign w:val="superscript"' in doc_xml
        assert '[1] Reference entry.' in doc_xml

    def test_stage24_postprocess_docx_un_numbers_references_heading_and_scales_figures(
        self,
        tmp_path: Path,
    ) -> None:
        docx_path = tmp_path / "paper_repaired.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="4242"/></w:numPr></w:pPr><w:r><w:t>References</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="CaptionedFigure"/></w:pPr><w:r><w:drawing><wp:inline><wp:extent cx="3000000" cy="1500000"/><wp:docPr descr="fig"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="3000000" cy="1500000"/></a:xfrm></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>"""
        styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style>
  <w:style w:type="paragraph" w:styleId="CaptionedFigure"><w:name w:val="Captioned Figure"/></w:style>
  <w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
  <w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/></w:style>
</w:styles>"""
        numbering_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>"""
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

        with ZipFile(docx_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml)
            zf.writestr("word/numbering.xml", numbering_xml)

        stage24_mod._postprocess_editorial_docx(docx_path)

        with ZipFile(docx_path) as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")

        refs_idx = doc_xml.find(">References<")
        assert refs_idx != -1
        refs_window = doc_xml[max(0, refs_idx - 200):refs_idx + 200]
        assert "<w:numPr>" not in refs_window
        assert 'cx="5800000"' in doc_xml

    def test_stage24_hard_fails_when_codex_cli_is_unavailable(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text(
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "As shown in Figure 3, PACT leads on F1 while GraphCodeBERT remains close.\n\n"
                "This paragraph continues discussing the regime-level comparison and why the top methods are close.\n\n"
                "Additional unrelated narrative about leakage and future work that should stay after the figure.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "**Figure 3.** Main F1 comparison across methods.\n"
            ),
            encoding="utf-8",
        )
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        charts = stage22 / "charts"
        charts.mkdir()
        (charts / "fig_main_f1_results.png").write_bytes(b"mainf1")

        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text(
            (stage22 / "paper_final.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (stage23 / "verification_report.json").write_text(
            json.dumps({"summary": {"verified": 10, "total": 10}}),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(stage24_mod, "_resolve_editorial_codex_binary", lambda _cfg: None)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert result.status is StageStatus.FAILED
        assert "codex" in (result.error or "").lower()

        assessment = json.loads(
            (stage_dir / "editorial_final_assessment.json").read_text(encoding="utf-8")
        )
        assert assessment["status"] == "fail"
        assert assessment["remaining_high_severity_issues"] >= 1

    def test_stage24_invokes_codex_without_nested_sandbox(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        seen: list[list[str]] = []

        def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
            cmd = cast(list[str], kwargs["args"] if "args" in kwargs else args[0])
            seen.append(cmd)
            cwd = cast(Path, kwargs["cwd"])
            (cwd / "paper_repaired.md").write_text("# Test Paper\n\n## Results\n\nRepaired.\n", encoding="utf-8")
            (cwd / "codex_review.json").write_text(
                json.dumps(
                    {
                        "summary": "No major issues remain.",
                        "issues": [],
                        "remaining_risks": [],
                        "should_continue": False,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout='{"type":"thread.started"}\n', stderr="")

        monkeypatch.setattr(stage24_mod, "_resolve_editorial_codex_binary", lambda _cfg: "/usr/bin/codex")
        monkeypatch.setattr(stage24_mod.subprocess, "run", _fake_run)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        cmd = next(cmd for cmd in seen if cmd[:2] == ["bash", "-lc"])
        assert cmd[:2] == ["bash", "-lc"]
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd[2]
        assert "--sandbox" not in cmd[2]
        assert result.status is StageStatus.DONE

    def test_stage24_invokes_codex_via_shell_wrapper(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        seen: list[list[str]] = []

        def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
            cmd = cast(list[str], kwargs["args"] if "args" in kwargs else args[0])
            seen.append(cmd)
            cwd = cast(Path, kwargs["cwd"])
            (cwd / "paper_repaired.md").write_text("# Test Paper\n\n## Results\n\nRepaired.\n", encoding="utf-8")
            (cwd / "codex_review.json").write_text(
                json.dumps(
                    {
                        "summary": "No major issues remain.",
                        "issues": [],
                        "remaining_risks": [],
                        "should_continue": False,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout='{"type":"thread.started"}\n', stderr="")

        monkeypatch.setattr(stage24_mod, "_resolve_editorial_codex_binary", lambda _cfg: "/usr/bin/codex")
        monkeypatch.setattr(stage24_mod.subprocess, "run", _fake_run)

        rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        cmd = next(cmd for cmd in seen if cmd[:2] == ["bash", "-lc"])
        assert cmd[:2] == ["bash", "-lc"]
        assert "cd " in cmd[2]
        assert '"/usr/bin/codex" exec' in cmd[2]
        assert "-C ." in cmd[2]

    def test_stage24_shell_wrapper_uses_absolute_workspace_path(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")

        rel_stage_dir = Path("relative-stage-24")
        if rel_stage_dir.exists():
            import shutil
            shutil.rmtree(rel_stage_dir)
        rel_stage_dir.mkdir(parents=True, exist_ok=True)

        seen: list[list[str]] = []

        def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
            cmd = cast(list[str], kwargs["args"] if "args" in kwargs else args[0])
            seen.append(cmd)
            cwd = cast(Path, kwargs["cwd"])
            (cwd / "paper_repaired.md").write_text("# Test Paper\n\n## Results\n\nRepaired.\n", encoding="utf-8")
            (cwd / "codex_review.json").write_text(
                json.dumps(
                    {
                        "summary": "No major issues remain.",
                        "issues": [],
                        "remaining_risks": [],
                        "should_continue": False,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout='{"type":"thread.started"}\n', stderr="")

        monkeypatch.setattr(stage24_mod, "_resolve_editorial_codex_binary", lambda _cfg: "/usr/bin/codex")
        monkeypatch.setattr(stage24_mod.subprocess, "run", _fake_run)

        try:
            rc_executor._execute_final_editorial_repair(
                rel_stage_dir, run_dir, rc_config, adapters, llm=None
            )
        finally:
            import shutil
            shutil.rmtree(rel_stage_dir, ignore_errors=True)

        cmd = next(cmd for cmd in seen if cmd[:2] == ["bash", "-lc"])
        shell_cmd = cmd[2]
        assert 'cd "' in shell_cmd
        cd_target = shell_cmd.split('cd "', 1)[1].split('"', 1)[0]
        assert cd_target.startswith("/")

    def test_stage24_accepts_explanations_after_caption(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text(
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "![Main](charts/fig_main_f1_results.png)\n\n"
                "*Figure 1. Main comparison.*\n\n"
                "This paragraph explains what Figure 1 shows and why the methods remain close.\n"
            ),
            encoding="utf-8",
        )
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text(
            (stage22 / "paper_final.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            _ = stage_dir_arg, run_dir_arg, config_arg
            return FakeEditorialLoopResult(
                success=True,
                markdown=source_markdown,
                review={"source": "stage-23/paper_final_verified.md", "initial_issue_count": 0, "issues": []},
                iterations=[],
                assessment={"status": "pass", "remaining_issue_count": 0, "remaining_high_severity_issues": 0},
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assessment = json.loads(
            (stage_dir / "editorial_final_assessment.json").read_text(encoding="utf-8")
        )
        assert result.status is StageStatus.DONE
        assert assessment["status"] == "pass"

    def test_stage24_failure_reports_assessment_reason(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Test Paper\n\n## Results\n\nText.\n", encoding="utf-8")

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            _ = stage_dir_arg, run_dir_arg, source_markdown, config_arg
            return FakeEditorialLoopResult(
                success=False,
                markdown="# Test Paper\n\n## Results\n\nText.\n",
                review={"source": "stage-23/paper_final_verified.md", "initial_issue_count": 1, "issues": [{"type": "missing_explanation", "severity": "high"}]},
                iterations=[{"iteration": 1, "action": "codex_editorial_rewrite", "changed": True}],
                assessment={"status": "fail", "remaining_issue_count": 1, "remaining_high_severity_issues": 1},
                error="assessment failed: 1 high-severity figure issue remains",
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert result.status is StageStatus.FAILED
        assert "high-severity" in (result.error or "")

    def test_stage24_audit_accepts_italic_figure_caption_blocks(self) -> None:
        blocks = rc_executor._safe_json_loads("{}", {})
        _ = blocks
        bundles = stage24_mod._extract_bundles(
            stage24_mod._split_blocks(
                "# T\n\n## Results\n\n"
                "Figure 5 introduces the leakage comparison.\n\n"
                "![Leakage](charts/fig_semantic_vs_fusion_scatter.png)\n\n"
                "*Figure 5. High detection performance coexists with near-perfect project-ID recoverability.*\n\n"
                "This local explanation interprets the axes and the leakage trade-off.\n"
            )
        )
        assert len(bundles) == 1
        assert bundles[0].caption_index is not None
        assert bundles[0].explanation_indices

    def test_stage24_loop_stages_bib_and_charts_before_compile(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text(
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "Figure 1 introduces the leakage comparison.\n\n"
                "![Leakage](charts/fig_semantic_vs_fusion_scatter.png)\n\n"
                "*Figure 1. Leakage remains high even when vulnerability detection is strong.*\n"
            ),
            encoding="utf-8",
        )
        (stage22 / "references.bib").write_text(
            "@article{foo2024,title={Foo},author={Bar}}\n",
            encoding="utf-8",
        )
        charts = stage22 / "charts"
        charts.mkdir()
        (charts / "fig_semantic_vs_fusion_scatter.png").write_bytes(b"scatter")

        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text(
            (stage22 / "paper_final.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(stage24_mod, "_resolve_editorial_codex_binary", lambda _cfg: "/usr/bin/codex")

        def _fake_codex_round(
            *, workspace: Path, config: RCConfig, iteration: int
        ) -> tuple[bool, str, dict[str, object]]:
            _ = config, iteration
            text = (workspace / "paper_repaired.md").read_text(encoding="utf-8")
            text += "\nThis paragraph explains the leakage trade-off more directly.\n"
            return (
                True,
                text,
                {
                    "summary": "Minor explanatory repair applied.",
                    "issues": [],
                    "remaining_risks": [],
                    "should_continue": False,
                },
            )

        seen_compile: dict[str, bool] = {}

        def _fake_compile(stage_dir_arg: Path, repaired_markdown: str, config_arg: RCConfig) -> tuple[list[str], list[str], bool]:
            _ = repaired_markdown, config_arg
            seen_compile["has_bib"] = (stage_dir_arg / "references.bib").exists()
            seen_compile["has_chart"] = (stage_dir_arg / "charts" / "fig_semantic_vs_fusion_scatter.png").exists()
            return [], [], True

        monkeypatch.setattr(stage24_mod, "_invoke_codex_editorial_round", _fake_codex_round)
        monkeypatch.setattr(stage24_mod, "_compile_editorial_tex", _fake_compile)

        result = stage24_mod._run_codex_editorial_loop(
            stage_dir,
            run_dir,
            (stage23 / "paper_final_verified.md").read_text(encoding="utf-8"),
            rc_config,
        )

        assert result.success is True
        assert seen_compile == {"has_bib": True, "has_chart": True}

    def test_stage24_publish_first_writes_codex_review_and_warns_on_minor_risks(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text(
            (
                "# Test Paper\n\n"
                "## Results\n\n"
                "PACT remains competitive.\n\n"
                "![Main F1](charts/fig_main_f1_results.png)\n\n"
                "*Figure 1. Main comparison across methods.*\n"
            ),
            encoding="utf-8",
        )
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        charts = stage22 / "charts"
        charts.mkdir()
        (charts / "fig_main_f1_results.png").write_bytes(b"mainf1")

        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text(
            (stage22 / "paper_final.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (stage23 / "verification_report.json").write_text(
            json.dumps({"summary": {"verified": 10, "total": 10}}),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        rc_config = RCConfig.from_dict(
            {
                **{
                    "project": {"name": rc_config.project.name, "mode": rc_config.project.mode},
                    "research": {
                        "topic": rc_config.research.topic,
                        "paper_title": rc_config.research.paper_title,
                        "domains": list(rc_config.research.domains),
                        "daily_paper_count": rc_config.research.daily_paper_count,
                        "quality_threshold": rc_config.research.quality_threshold,
                    },
                    "runtime": {
                        "timezone": rc_config.runtime.timezone,
                        "max_parallel_tasks": rc_config.runtime.max_parallel_tasks,
                        "approval_timeout_hours": rc_config.runtime.approval_timeout_hours,
                        "retry_limit": rc_config.runtime.retry_limit,
                    },
                    "notifications": {"channel": rc_config.notifications.channel},
                    "knowledge_base": {"backend": rc_config.knowledge_base.backend, "root": rc_config.knowledge_base.root},
                    "llm": {
                        "provider": rc_config.llm.provider,
                        "base_url": rc_config.llm.base_url,
                        "api_key_env": rc_config.llm.api_key_env,
                    },
                    "experiment": {
                        "mode": rc_config.experiment.mode,
                        "editorial_repair": {"mode": "publish_first", "max_iterations": 5},
                    },
                }
            },
            project_root=tmp_path,
            check_paths=False,
        )

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            _ = stage_dir_arg, run_dir_arg
            repaired = source_markdown.replace(
                "PACT remains competitive.\n\n",
                "PACT remains competitive on the primary metric.\n\n"
                "Figure 1 makes the ranking explicit and shows why the margin over the strongest baseline should be read as modest rather than decisive.\n\n",
            )
            return FakeEditorialLoopResult(
                success=True,
                markdown=repaired,
                review={
                    "source": "stage-23/paper_final_verified.md",
                    "initial_issue_count": 0,
                    "issues": [],
                    "codex_review": {
                        "issues": [
                            {"type": "missing_local_narrative", "severity": "medium"}
                        ],
                        "remaining_risks": ["minor phrasing polish remains"],
                        "should_continue": False,
                    },
                },
                iterations=[
                    {
                        "iteration": 1,
                        "action": "codex_editorial_review_and_rewrite",
                        "changed": True,
                    }
                ],
                assessment={
                    "status": "warn",
                    "remaining_issue_count": 0,
                    "remaining_high_severity_issues": 0,
                    "compile_clean": True,
                    "improved_vs_stage22": True,
                    "boundary_violations": [],
                    "codex_reported_remaining_risks": ["minor phrasing polish remains"],
                    "used_iterations": 1,
                },
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        review = json.loads((stage_dir / "editorial_review.json").read_text(encoding="utf-8"))
        assessment = json.loads(
            (stage_dir / "editorial_final_assessment.json").read_text(encoding="utf-8")
        )
        assert result.status is StageStatus.DONE
        assert review["codex_review"]["issues"][0]["type"] == "missing_local_narrative"
        assert assessment["status"] == "warn"
        assert assessment["improved_vs_stage22"] is True
        assert assessment["compile_clean"] is True

    def test_stage24_editorial_task_mentions_layout_polish_and_page_breaks(self) -> None:
        task = stage24_mod._build_editorial_task(
            issue_report=[],
            iteration=1,
            source_label="stage-24/paper_repaired.md",
            mode="publish_first",
        )

        assert "single-figure page" in task
        assert "awkward page breaks" in task
        assert "shrink a figure modestly" in task

    def test_stage24_workspace_prefers_existing_stage24_tex_pdf_for_resume(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper.tex").write_text("stage22-tex", encoding="utf-8")
        (stage22 / "compilation_quality.json").write_text("{}", encoding="utf-8")
        (stage22 / "paper_verification.json").write_text("{}", encoding="utf-8")
        (stage22 / "pdf_review.json").write_text("{}", encoding="utf-8")
        (stage22 / "references.bib").write_text("% refs", encoding="utf-8")
        charts = stage22 / "charts"
        charts.mkdir()
        (charts / "fig.png").write_bytes(b"png")

        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "verification_report.json").write_text("{}", encoding="utf-8")

        stage24 = run_dir / "stage-24"
        stage24.mkdir(parents=True, exist_ok=True)
        (stage24 / "paper_repaired.tex").write_text("stage24-tex", encoding="utf-8")
        (stage24 / "paper_repaired.pdf").write_bytes(b"%PDF-1.5")

        workspace = stage24_mod._prepare_codex_workspace(
            stage_dir=tmp_path / "fresh-stage-24",
            run_dir=run_dir,
            source_markdown="# T\n\nBody\n",
            issue_report=[],
            iteration=1,
            source_label="stage-24/paper_repaired.md",
            mode="publish_first",
        )

        assert (workspace / "paper_repaired.tex").read_text(encoding="utf-8") == "stage24-tex"
        assert (workspace / "paper_repaired.pdf").read_bytes() == b"%PDF-1.5"

    def test_stage24_prefers_existing_repaired_markdown_as_resume_source(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Stage22\n\nOld base.\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Stage23\n\nVerified base.\n", encoding="utf-8")
        stage24_prev = run_dir / "stage-24"
        stage24_prev.mkdir(parents=True, exist_ok=True)
        (stage24_prev / "paper_repaired.md").write_text("# Stage24\n\nContinue from repaired.\n", encoding="utf-8")
        (stage24_prev / "editorial_final_assessment.json").write_text(
            json.dumps({"status": "pass"}), encoding="utf-8"
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, object] = {}

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            _ = stage_dir_arg, run_dir_arg, config_arg
            seen["source_markdown"] = source_markdown
            return FakeEditorialLoopResult(
                success=True,
                markdown=source_markdown,
                review={"source": "stage-24/paper_repaired.md", "initial_issue_count": 0, "issues": []},
                iterations=[],
                assessment={"status": "pass", "remaining_issue_count": 0, "remaining_high_severity_issues": 0, "compile_clean": True, "improved_vs_stage22": True, "boundary_violations": [], "codex_reported_remaining_risks": [], "used_iterations": 1},
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert result.status is StageStatus.DONE
        assert cast(str, seen["source_markdown"]).startswith("# Stage24")

    def test_stage24_reuses_failed_repaired_markdown_before_stage23(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_editorial_pipeline(monkeypatch)
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# Stage22\n\nOld base.\n", encoding="utf-8")
        stage23 = run_dir / "stage-23"
        stage23.mkdir(parents=True, exist_ok=True)
        (stage23 / "paper_final_verified.md").write_text("# Stage23\n\nVerified base.\n", encoding="utf-8")
        stage24_prev = run_dir / "stage-24"
        stage24_prev.mkdir(parents=True, exist_ok=True)
        (stage24_prev / "paper_repaired.md").write_text("# Failed Stage24\n\nStill better repaired draft.\n", encoding="utf-8")
        (stage24_prev / "editorial_final_assessment.json").write_text(
            json.dumps({"status": "fail", "remaining_high_severity_issues": 1}),
            encoding="utf-8",
        )

        stage_dir = tmp_path / "run" / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, object] = {}

        def _fake_codex_loop(
            stage_dir_arg: Path,
            run_dir_arg: Path,
            source_markdown: str,
            config_arg: RCConfig,
        ) -> FakeEditorialLoopResult:
            _ = stage_dir_arg, run_dir_arg, config_arg
            seen["source_markdown"] = source_markdown
            return FakeEditorialLoopResult(
                success=True,
                markdown=source_markdown,
                review={"source": "stage-24/paper_repaired.md", "initial_issue_count": 0, "issues": []},
                iterations=[],
                assessment={"status": "pass", "remaining_issue_count": 0, "remaining_high_severity_issues": 0, "compile_clean": True, "improved_vs_stage22": True, "boundary_violations": [], "codex_reported_remaining_risks": [], "used_iterations": 1},
            )

        monkeypatch.setattr(stage24_mod, "_run_codex_editorial_loop", _fake_codex_loop)

        result = rc_executor._execute_final_editorial_repair(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert result.status is StageStatus.DONE
        assert cast(str, seen["source_markdown"]).startswith("# Failed Stage24")

    def test_stage24_publish_first_allows_removing_degraded_note_with_numbers(self) -> None:
        original = (
            "# Title\n\n"
            "## Abstract\n\n"
            "Main abstract text.\n\n"
            "> **Note:** This paper was produced in degraded mode. Quality gate score (3.2/4.0) "
            "was below threshold. Unverified numerical results in tables have been replaced with "
            "`---` and require independent verification.\n\n"
            "## Results\n\n"
            "PACT remains competitive.\n"
        )
        repaired = (
            "# Title\n\n"
            "## Abstract\n\n"
            "Main abstract text.\n\n"
            "## Results\n\n"
            "PACT remains competitive.\n"
        )

        violations = stage24_mod._detect_boundary_violations(
            original,
            repaired,
            mode="publish_first",
        )

        assert violations == []

    def test_stage24_balanced_allows_prose_number_changes(self) -> None:
        original = "# T\n\nMetric text reports 0.9299 F1 in prose.\n"
        repaired = "# T\n\nMetric text reports 0.9301 F1 in prose.\n"

        violations = stage24_mod._detect_boundary_violations(
            original,
            repaired,
            mode="balanced",
        )

        assert violations == []

    def test_stage24_still_blocks_citation_key_changes(self) -> None:
        original = "# T\n\nText with citation [chen2024deep].\n"
        repaired = "# T\n\nText with citation [kumar2024large].\n"

        violations = stage24_mod._detect_boundary_violations(
            original,
            repaired,
            mode="publish_first",
        )

        assert violations == ["citation_keys_changed"]

    def test_stage24_ignores_latex_float_option_brackets_in_boundary_check(self) -> None:
        original = "# T\n\nText with citation [chen2024deep].\n"
        repaired = (
            "# T\n\nText with citation [chen2024deep].\n\n"
            "\\begin{figure}[t]\n\\centering\n\\end{figure}\n"
        )

        violations = stage24_mod._detect_boundary_violations(
            original,
            repaired,
            mode="publish_first",
        )

        assert violations == []

    def test_stage24_audit_compiled_layout_detects_float_before_section(self, tmp_path: Path) -> None:
        stage_dir = tmp_path / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "paper_repaired.tex").write_text(
            (
                "\\begin{figure}[!htbp]\n"
                "\\centering\n"
                "\\caption{Leakage scatter}\n"
                "\\label{fig:leakage_scatter}\n"
                "\\end{figure}\n\n"
                "\\section{Discussion}\n"
            ),
            encoding="utf-8",
        )

        issues = stage24_mod._audit_compiled_layout(stage_dir)

        assert issues
        assert issues[0]["type"] == "awkward_float_layout"
        assert issues[0]["severity"] == "high"
        assert issues[0]["target_kind"] == "figure"
        assert issues[0]["target_hint"] == "fig:leakage_scatter"
        assert issues[0]["section"] == "Discussion"

    def test_stage24_audit_compiled_layout_prefers_last_float_before_section(self, tmp_path: Path) -> None:
        stage_dir = tmp_path / "stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "paper_repaired.tex").write_text(
            (
                "\\begin{figure}[!htbp]\n"
                "\\caption{Overview}\n"
                "\\label{fig:overview_of_pact}\n"
                "\\end{figure}\n\n"
                "\\section{Results}\n"
                "Text.\n\n"
                "\\begin{table}[ht]\n"
                "\\caption{Per-regime table}\n"
                "\\label{tab:3}\n"
                "\\end{table}\n\n"
                "Short transition.\n\n"
                "\\begin{figure}[!htbp]\n"
                "\\caption{Leakage scatter}\n"
                "\\label{fig:mean_f1_versus_project_id_prob}\n"
                "\\end{figure}\n\n"
                "\\section{Discussion}\n"
            ),
            encoding="utf-8",
        )

        issues = stage24_mod._audit_compiled_layout(stage_dir)

        assert issues
        assert issues[0]["target_hint"] == "fig:mean_f1_versus_project_id_prob"
        assert issues[0]["section"] == "Discussion"

    def test_stage24_publish_first_continues_when_compiled_layout_issue_remains(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# T\n\nBody.\n", encoding="utf-8")
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        (stage22 / "references.bib").write_text("% refs", encoding="utf-8")

        rc_config = RCConfig.from_dict(
            {
                "project": {"name": rc_config.project.name, "mode": rc_config.project.mode},
                "research": {
                    "topic": rc_config.research.topic,
                    "paper_title": rc_config.research.paper_title,
                    "domains": list(rc_config.research.domains),
                    "daily_paper_count": rc_config.research.daily_paper_count,
                    "quality_threshold": rc_config.research.quality_threshold,
                },
                "runtime": {
                    "timezone": rc_config.runtime.timezone,
                    "max_parallel_tasks": rc_config.runtime.max_parallel_tasks,
                    "approval_timeout_hours": rc_config.runtime.approval_timeout_hours,
                    "retry_limit": rc_config.runtime.retry_limit,
                },
                "notifications": {"channel": rc_config.notifications.channel},
                "knowledge_base": {"backend": rc_config.knowledge_base.backend, "root": rc_config.knowledge_base.root},
                "llm": {
                    "provider": rc_config.llm.provider,
                    "base_url": rc_config.llm.base_url,
                    "api_key_env": rc_config.llm.api_key_env,
                },
                "experiment": {
                    "mode": rc_config.experiment.mode,
                    "editorial_repair": {"mode": "publish_first", "max_iterations": 3},
                },
            },
            project_root=tmp_path,
            check_paths=False,
        )

        call_count = {"n": 0}

        def _fake_codex_round(*, workspace: Path, config: RCConfig, iteration: int):
            _ = workspace, config
            call_count["n"] += 1
            return (
                True,
                f"# T\n\nIteration {iteration}\n",
                {
                    "summary": "review",
                    "issues": [],
                    "remaining_risks": [],
                    "should_continue": False,
                },
            )

        def _fake_compile(stage_dir_arg: Path, repaired_markdown: str, config_arg: RCConfig):
            _ = repaired_markdown, config_arg
            tex = (
                "\\begin{figure}[!htbp]\n\\caption{Leakage scatter}\n\\end{figure}\n\n\\section{Discussion}\n"
                if call_count["n"] == 1
                else "\\section{Discussion}\nText.\n"
            )
            (stage_dir_arg / "paper_repaired.tex").write_text(tex, encoding="utf-8")
            return ["paper_repaired.tex"], ["stage-24/paper_repaired.tex"], True

        monkeypatch.setattr(stage24_mod, "_invoke_codex_editorial_round", _fake_codex_round)
        monkeypatch.setattr(stage24_mod, "_compile_editorial_tex", _fake_compile)

        stage_dir = tmp_path / "fresh-stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = stage24_mod._run_codex_editorial_loop(
            stage_dir,
            run_dir,
            "# T\n\nBody.\n",
            rc_config,
        )

        assert result.success is True
        assert call_count["n"] == 2
        assert result.review["final_issues"] == []

    def test_stage24_final_outputs_reflect_compiled_layout_issue_details(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage22 = run_dir / "stage-22"
        stage22.mkdir(parents=True, exist_ok=True)
        (stage22 / "paper_final.md").write_text("# T\n\nBody.\n", encoding="utf-8")
        (stage22 / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
        (stage22 / "references.bib").write_text("% refs", encoding="utf-8")

        rc_config = RCConfig.from_dict(
            {
                "project": {"name": rc_config.project.name, "mode": rc_config.project.mode},
                "research": {
                    "topic": rc_config.research.topic,
                    "paper_title": rc_config.research.paper_title,
                    "domains": list(rc_config.research.domains),
                    "daily_paper_count": rc_config.research.daily_paper_count,
                    "quality_threshold": rc_config.research.quality_threshold,
                },
                "runtime": {
                    "timezone": rc_config.runtime.timezone,
                    "max_parallel_tasks": rc_config.runtime.max_parallel_tasks,
                    "approval_timeout_hours": rc_config.runtime.approval_timeout_hours,
                    "retry_limit": rc_config.runtime.retry_limit,
                },
                "notifications": {"channel": rc_config.notifications.channel},
                "knowledge_base": {"backend": rc_config.knowledge_base.backend, "root": rc_config.knowledge_base.root},
                "llm": {
                    "provider": rc_config.llm.provider,
                    "base_url": rc_config.llm.base_url,
                    "api_key_env": rc_config.llm.api_key_env,
                },
                "experiment": {
                    "mode": rc_config.experiment.mode,
                    "editorial_repair": {"mode": "publish_first", "max_iterations": 1},
                },
            },
            project_root=tmp_path,
            check_paths=False,
        )

        def _fake_codex_round(*, workspace: Path, config: RCConfig, iteration: int):
            _ = workspace, config, iteration
            return (
                True,
                "# T\n\nBody.\n",
                {
                    "summary": "review",
                    "issues": [],
                    "remaining_risks": [],
                    "should_continue": False,
                },
            )

        def _fake_compile(stage_dir_arg: Path, repaired_markdown: str, config_arg: RCConfig):
            _ = repaired_markdown, config_arg
            (stage_dir_arg / "paper_repaired.tex").write_text(
                "\\begin{figure}[!htbp]\n"
                "\\caption{Leakage scatter}\n"
                "\\label{fig:leakage_scatter}\n"
                "\\end{figure}\n\n"
                "\\section{Discussion}\n",
                encoding="utf-8",
            )
            return ["paper_repaired.tex"], ["stage-24/paper_repaired.tex"], True

        monkeypatch.setattr(stage24_mod, "_invoke_codex_editorial_round", _fake_codex_round)
        monkeypatch.setattr(stage24_mod, "_compile_editorial_tex", _fake_compile)

        stage_dir = tmp_path / "fresh-stage-24"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = stage24_mod._run_codex_editorial_loop(
            stage_dir,
            run_dir,
            "# T\n\nBody.\n",
            rc_config,
        )

        assert result.success is False
        assert result.assessment["remaining_issue_types"] == ["awkward_float_layout"]
        assert "awkward page breaks" in result.assessment["remaining_issue_notes"][0]
        assert result.review["final_issues"][0]["type"] == "awkward_float_layout"


# ── P1-1: Topic keyword extraction tests ──


class TestExtractTopicKeywords:
    def test_basic_extraction(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Agent-based Reinforcement Learning for Automated Scientific Discovery"
        )
        assert "agent-based" in keywords
        assert "reinforcement" in keywords
        assert "learning" in keywords
        assert "automated" in keywords
        assert "scientific" in keywords
        assert "discovery" in keywords
        # Stop words excluded
        # Stop words excluded
        assert "for" not in keywords

    def test_includes_domain_keywords(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Neural network pruning", domains=("ml", "optimization")
        )
        assert "neural" in keywords
        assert "network" in keywords
        assert "pruning" in keywords
        assert "ml" in keywords
        assert "optimization" in keywords

    def test_deduplication(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Learning to learn meta-learning", domains=("learning",)
        )
        assert keywords.count("learning") == 1

    def test_empty_topic(self) -> None:
        keywords = rc_executor._extract_topic_keywords("")
        assert keywords == []


# ── P1-2: Topic constraint block test ──


class TestTopicConstraintBlock:
    def test_contains_topic(self) -> None:
        block = rc_executor._topic_constraint_block("Transformer attention for time series")
        assert "Transformer attention for time series" in block

    def test_contains_prohibition(self) -> None:
        block = rc_executor._topic_constraint_block("anything")
        assert "PROHIBITED" in block
        assert "environment" in block.lower()
        assert "infrastructure" in block.lower()

    def test_hard_constraint_markers(self) -> None:
        block = rc_executor._topic_constraint_block("test")
        assert "HARD TOPIC CONSTRAINT" in block
        assert "END CONSTRAINT" in block


# ── Multi-perspective debate tests ──


class TestParseDecision:
    def test_proceed_default(self) -> None:
        assert rc_executor._parse_decision("Some random text") == "proceed"

    def test_proceed_explicit(self) -> None:
        text = "## Decision\nPROCEED\n## Justification\nGood results."
        assert rc_executor._parse_decision(text) == "proceed"

    def test_pivot_detected(self) -> None:
        text = "## Decision\nPIVOT\n## Justification\nHypotheses flawed."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_refine_detected(self) -> None:
        text = "## Decision\nREFINE\n## Justification\nNeed more tuning."
        assert rc_executor._parse_decision(text) == "refine"

    def test_pivot_case_insensitive(self) -> None:
        text = "## Decision\npivot\n## Justification\nBad approach."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_pivot_takes_priority_over_proceed(self) -> None:
        text = "## Decision\nPIVOT\nWe should not PROCEED."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_decision_in_body_not_heading(self) -> None:
        text = "The results suggest we should PIVOT to a new approach."
        assert rc_executor._parse_decision(text) == "pivot"


class TestResearchDecisionStructured:
    def test_decision_produces_structured_json(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nResults ok.")
        fake_llm = FakeLLMClient("## Decision\nPROCEED\n## Justification\nGood.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "proceed"
        assert "decision_structured.json" in result.artifacts
        import json
        data = json.loads((stage_dir / "decision_structured.json").read_text())
        assert data["decision"] == "proceed"

    def test_pivot_decision_from_llm(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nBad results.")
        fake_llm = FakeLLMClient("## Decision\nPIVOT\n## Justification\nFlawed.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "pivot"

    def test_no_llm_defaults_to_proceed(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.decision == "proceed"


class TestMultiPerspectiveGenerate:
    def test_generates_all_perspectives(self, tmp_path: Path) -> None:
        roles = {
            "role_a": {"system": "You are A.", "user": "Do A for {topic}."},
            "role_b": {"system": "You are B.", "user": "Do B for {topic}."},
        }
        fake_llm = FakeLLMClient("perspective output")
        perspectives_dir = tmp_path / "perspectives"
        result = rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "test"}, perspectives_dir
        )
        assert set(result.keys()) == {"role_a", "role_b"}
        assert (perspectives_dir / "role_a.md").exists()
        assert (perspectives_dir / "role_b.md").exists()
        assert len(fake_llm.calls) == 2

    def test_saves_perspective_content(self, tmp_path: Path) -> None:
        roles = {"critic": {"system": "Be critical.", "user": "Criticize {topic}."}}
        fake_llm = FakeLLMClient("critical analysis here")
        perspectives_dir = tmp_path / "perspectives"
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "ml"}, perspectives_dir
        )
        content = (perspectives_dir / "critic.md").read_text()
        assert content == "critical analysis here"

    def test_renders_variables_in_prompts(self, tmp_path: Path) -> None:
        roles = {"r1": {"system": "Sys for {topic}.", "user": "User for {topic}."}}
        fake_llm = FakeLLMClient("ok")
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "RL"}, tmp_path / "p"
        )
        call = fake_llm.calls[0]
        assert "RL" in call[0]["content"]


class TestSynthesizePerspectives:
    def test_combines_perspectives(self) -> None:
        fake_llm = FakeLLMClient("synthesized result")
        pm = rc_executor.PromptManager()
        perspectives = {"innovator": "idea A", "contrarian": "idea B"}
        result = rc_executor._synthesize_perspectives(
            fake_llm, perspectives, "hypothesis_synthesize", pm
        )
        assert result == "synthesized result"
        # Check the user prompt contained both perspectives
        call_content = fake_llm.calls[0][0]["content"]
        assert "innovator" in call_content
        assert "contrarian" in call_content


class TestHypothesisGenDebate:
    def test_hypothesis_gen_with_llm_creates_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        fake_llm = FakeLLMClient("## H1\nTest hypothesis")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        perspectives_dir = stage_dir / "perspectives"
        assert perspectives_dir.exists()
        # Should have 3 perspective files (innovator, pragmatist, contrarian)
        perspective_files = list(perspectives_dir.glob("*.md"))
        assert len(perspective_files) == 3

    def test_hypothesis_gen_without_llm_no_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        # No perspectives directory when no LLM
        assert not (stage_dir / "perspectives").exists()


class TestResultAnalysisDebate:
    def test_result_analysis_with_llm_creates_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 1, "goal.md", "# Goal\nTest")
        _write_prior_artifact(run_dir, 8, "hypotheses.md", "# H1\nTest")
        fake_llm = FakeLLMClient("## Analysis\nResults look good.")
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        perspectives_dir = stage_dir / "perspectives"
        assert perspectives_dir.exists()
        # Should have 3 perspective files (optimist, skeptic, methodologist)
        perspective_files = list(perspectives_dir.glob("*.md"))
        assert len(perspective_files) == 3

    def test_result_analysis_without_llm_no_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        assert not (stage_dir / "perspectives").exists()


class TestParseMetricsFromStdout:
    """Tests for _parse_metrics_from_stdout() helper."""

    def test_parses_simple_name_value(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "loss: 0.0042\naccuracy: 0.95"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["loss"] == pytest.approx(0.0042)
        assert metrics["accuracy"] == pytest.approx(0.95)

    def test_parses_compound_names(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Stochastic) cumulative_regret: 361.9233\nEXP3 (Adversarial) total_rewards: 13368.4811"
        metrics = _parse_metrics_from_stdout(stdout)
        assert "UCB (Stochastic) cumulative_regret" in metrics
        assert metrics["UCB (Stochastic) cumulative_regret"] == pytest.approx(361.9233)

    def test_ignores_non_numeric_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "Running experiment...\nloss: 0.5\nDone."
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 1
        assert metrics["loss"] == pytest.approx(0.5)

    def test_empty_stdout_returns_empty_dict(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        assert _parse_metrics_from_stdout("") == {}

    def test_handles_negative_values(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Adversarial) cumulative_regret: -3877.5323"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["UCB (Adversarial) cumulative_regret"] == pytest.approx(-3877.5323)

    def test_filters_log_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = (
            "Running experiments for support set size: 1\n"
            "Loading model weights: 42\n"
            "Training epoch: 5\n"
            "loss: 0.123\n"
            "accuracy: 0.95\n"
        )
        metrics = _parse_metrics_from_stdout(stdout)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert len(metrics) == 2  # log lines should be excluded

    def test_filters_long_name_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "this is a very long status message that should not be a metric: 42\n"
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 0


class TestDetectRuntimeIssues:
    """Tests for _detect_runtime_issues() helper."""

    def _make_sandbox_result(
        self,
        metrics: dict | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        from types import SimpleNamespace

        return SimpleNamespace(
            metrics=metrics or {},
            stdout=stdout,
            stderr=stderr,
            returncode=0,
            elapsed_sec=1.0,
            timed_out=False,
        )

    def test_no_issues_returns_empty_string(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": 0.5}, stdout="loss: 0.5")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_detects_nan_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("nan")})
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "loss" in result

    def test_detects_inf_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("inf")})
        result = rc_executor._detect_runtime_issues(r)
        assert "Inf" in result

    def test_detects_nan_in_stdout(self) -> None:
        r = self._make_sandbox_result(stdout="accuracy: nan\nloss: 0.5")
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result or "nan" in result

    def test_detects_runtime_warning_in_stderr(self) -> None:
        stderr = (
            "optimizers.py:76: RuntimeWarning: invalid value encountered in divide\n"
            "  directions = np.vstack((directions[1:], new_direction / norm))\n"
        )
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "RuntimeWarning" in result
        assert "invalid value" in result

    def test_detects_division_error_in_stderr(self) -> None:
        stderr = "ZeroDivisionError: division by zero\n"
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "Error" in result

    def test_ignores_benign_stderr(self) -> None:
        # Non-warning stderr should be ignored
        r = self._make_sandbox_result(stderr="Loading module...\nDone.\n")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_combined_nan_and_stderr(self) -> None:
        r = self._make_sandbox_result(
            metrics={"accuracy": float("nan")},
            stderr="RuntimeWarning: invalid value\n",
        )
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "RuntimeWarning" in result

    def test_detects_dummy_metric_identical_values(self) -> None:
        stdout = (
            "UCB (Stochastic) convergence_rate: 1.0000\n"
            "UCB (Adversarial) convergence_rate: 1.0000\n"
            "Thompson (Stochastic) convergence_rate: 1.0000\n"
            "Thompson (Adversarial) convergence_rate: 1.0000\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" in result
        assert "convergence_rate" in result

    def test_no_dummy_metric_when_values_differ(self) -> None:
        stdout = (
            "UCB (Stochastic) regret: 78.5\n"
            "Thompson (Stochastic) regret: 121.0\n"
            "EpsilonGreedy (Stochastic) regret: 42.1\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" not in result


class TestRemoveBibtexEntries:
    """Tests for _remove_bibtex_entries() helper."""

    def test_removes_specified_keys(self) -> None:
        bib = (
            '@article{smith2024,\n  title={Good Paper},\n  author={Smith},\n}\n\n'
            '@article{venus2024,\n  title={Venus Exploration},\n  author={NASA},\n}\n'
        )
        result = rc_executor._remove_bibtex_entries(bib, {"venus2024"})
        assert "smith2024" in result
        assert "venus2024" not in result

    def test_keeps_all_when_no_match(self) -> None:
        bib = '@article{smith2024,\n  title={Paper},\n}\n'
        result = rc_executor._remove_bibtex_entries(bib, {"other_key"})
        assert "smith2024" in result

    def test_empty_bib(self) -> None:
        assert rc_executor._remove_bibtex_entries("", {"key"}) == ""


class TestRemoveCitationsFromText:
    """Tests for _remove_citations_from_text() helper."""

    def test_removes_latex_cite(self) -> None:
        text = r"As shown in \cite{venus2024}, the results are..."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result
        assert "results are" in result

    def test_removes_markdown_cite(self) -> None:
        text = "Prior work [venus2024] explored this topic."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result

    def test_cleans_multi_cite_comma(self) -> None:
        text = r"\cite{good2024,venus2024}"
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert r"\cite{good2024}" in result


class TestCollectRawExperimentMetrics:
    """Tests for _collect_raw_experiment_metrics() helper."""

    def test_returns_empty_when_no_runs(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        block, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert block == ""
        assert not has_parsed

    def test_extracts_metrics_from_stdout(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {},
            "stdout": "UCB regret: 361.92\nThompson regret: 576.24\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "361.92" in result
        assert "576.24" in result
        assert "1 run(s)" in result
        assert not has_parsed

    def test_extracts_from_metrics_dict(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {"metrics": {"loss": 0.042, "accuracy": 0.95}, "stdout": ""}
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "loss" in result
        assert "0.042" in result
        assert has_parsed

    def test_deduplicates_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {"loss": 0.5},
            "stdout": "loss: 0.5\nloss: 0.5\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, _ = rc_executor._collect_raw_experiment_metrics(run_dir)
        # "loss: 0.5" should appear only once (deduplicated)
        assert result.count("loss: 0.5") == 1


class TestCollectExperimentEvidence:
    """Tests for _collect_experiment_evidence() helper."""

    def test_returns_empty_when_no_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert rc_executor._collect_experiment_evidence(run_dir) == ""

    def test_includes_main_py_code(self, run_dir: Path) -> None:
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "main.py" in result
        assert "hello" in result

    def test_includes_run_metrics(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}, "elapsed_sec": 3.2}),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "loss" in result
        assert "0.5" in result

    def test_includes_stderr_excerpt(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "metrics": {"loss": 0.5},
                "stderr": "RuntimeWarning: divide by zero",
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "divide by zero" in result

    def test_includes_refinement_summary(self, run_dir: Path) -> None:
        refine_dir = run_dir / "stage-13"
        refine_dir.mkdir(parents=True, exist_ok=True)
        (refine_dir / "refinement_log.json").write_text(
            json.dumps({
                "iterations": [{"iteration": 1}, {"iteration": 2}],
                "converged": True,
                "stop_reason": "no_improvement_for_2_iterations",
                "best_metric": 0.3,
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "iterations_executed" in result
        assert "2" in result

    def test_includes_actual_trial_count(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}}), encoding="utf-8"
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "1 time(s)" in result
        assert "CRITICAL" in result


class TestWritePaperSections:
    """Tests for _write_paper_sections() multi-call writing."""

    def test_produces_three_part_draft(self) -> None:
        call_count = {"n": 0}
        parts = [
            "# Test Title\n\n## Abstract\nTest abstract.\n\n## Introduction\nTest intro.\n\n## Related Work\nTest related.",
            "## Method\nTest method.\n\n## Experiments\nTest experiments.",
            "## Results\nTest results.\n\n## Discussion\nTest discussion.\n\n## Limitations\nTest limits.\n\n## Conclusion\nTest conclusion.",
        ]

        class MultiCallLLM:
            def __init__(self):
                self.calls: list = []

            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                from researchclaw.llm.client import LLMResponse
                idx = len(self.calls) - 1
                return LLMResponse(content=parts[min(idx, 2)], model="fake")

        llm = MultiCallLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        draft = rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Test outline",
        )

        assert llm.calls is not None
        assert len(llm.calls) == 3
        assert "## Abstract" in draft
        assert "## Method" in draft
        assert "## Results" in draft
        assert "## Conclusion" in draft

    def test_each_call_receives_prior_context(self) -> None:
        class ContextTrackingLLM:
            def __init__(self):
                self.user_prompts: list[str] = []

            def chat(self, messages, **kwargs):
                for m in messages:
                    if m.get("role") == "user":
                        self.user_prompts.append(m["content"])
                from researchclaw.llm.client import LLMResponse
                return LLMResponse(content="## Section\nContent here.", model="fake")

        llm = ContextTrackingLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Outline",
        )

        assert len(llm.user_prompts) == 3
        # Call 2 and 3 should contain "sections written so far"
        assert "sections written so far" in llm.user_prompts[1]
        assert "completing a paper" in llm.user_prompts[2]


class TestLoadHardwareProfile:
    """Tests for _load_hardware_profile()."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "run"
        d.mkdir()
        return d

    def test_loads_valid_profile(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        profile = {"has_gpu": True, "gpu_type": "mps", "tier": "limited"}
        (stage / "hardware_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        result = rc_executor._load_hardware_profile(run_dir)
        assert result is not None
        assert result["gpu_type"] == "mps"

    def test_returns_none_when_missing(self, run_dir: Path) -> None:
        assert rc_executor._load_hardware_profile(run_dir) is None

    def test_returns_none_on_invalid_json(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        (stage / "hardware_profile.json").write_text("not json", encoding="utf-8")
        assert rc_executor._load_hardware_profile(run_dir) is None


class TestExpandSearchQueries:
    """Tests for _expand_search_queries()."""

    def test_adds_broader_queries(self) -> None:
        queries = ["gradient descent optimization algorithms"]
        topic = "Comparing gradient descent optimization algorithms on benchmark functions"
        result = rc_executor._expand_search_queries(queries, topic)
        assert len(result) > len(queries)

    def test_deduplicates(self) -> None:
        queries = ["gradient descent survey"]
        topic = "gradient descent optimization"
        result = rc_executor._expand_search_queries(queries, topic)
        lowered = [q.lower().strip() for q in result]
        assert len(lowered) == len(set(lowered))

    def test_preserves_original_queries(self) -> None:
        queries = ["query A", "query B"]
        topic = "some research topic about machine learning methods"
        result = rc_executor._expand_search_queries(queries, topic)
        assert result[0] == "query A"
        assert result[1] == "query B"

    def test_adds_survey_benchmark_variants(self) -> None:
        queries = ["deep learning"]
        topic = "deep learning for image classification with limited data"
        result = rc_executor._expand_search_queries(queries, topic)
        has_survey = any("survey" in q.lower() for q in result)
        has_benchmark = any("benchmark" in q.lower() for q in result)
        assert has_survey
        assert has_benchmark


# ── R4-1: Experiment Budget Guard Tests ──────────────────────────────


class TestComputeBudgetBlock:
    """Test compute_budget prompt block injection (R4-1a)."""

    def test_compute_budget_block_exists_in_prompt_manager(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "time_budget_sec" in block or "Compute Budget" in block

    def test_compute_budget_injected_into_code_generation(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        import sys

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "optimizer comparison",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 60,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # Write exp_plan prior artifact
        _write_prior_artifact(run_dir, 10, "exp_plan.yaml", "objectives: test")

        # Capture what the LLM receives
        llm = FakeLLMClient(
            "```filename:main.py\nimport numpy as np\nprint('best_loss: 0.1')\n```"
        )
        stage_dir = run_dir / "stage-11"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_code_generation(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received compute budget info in some call
        # (may be first call in legacy mode, or second call with CodeAgent)
        assert len(llm.calls) >= 1
        all_user_msgs = " ".join(
            call[-1]["content"] for call in llm.calls if call
        )
        assert "60" in all_user_msgs or "Compute Budget" in all_user_msgs


class TestPartialTimeoutStatus:
    """Test partial status for timed-out experiments with data (R4-1c)."""

    def test_timed_out_with_metrics_sets_partial_status(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        import sys

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test topic",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 2,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # Write experiment code that prints some metrics then sleeps
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text(
            "import time, sys\n"
            "print('best_loss: 0.5', flush=True)\n"
            "sys.stdout.flush()\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        run_file = stage_dir / "runs" / "run-1.json"
        assert run_file.exists()
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        # Should be "partial" since metrics were captured before timeout
        assert payload["timed_out"] is True
        # Status should be "partial" if metrics captured, "failed" if not
        if payload["metrics"]:
            assert payload["status"] == "partial"
        else:
            # Subprocess stdout may not flush before kill on some platforms
            assert payload["status"] == "failed"


class TestTimeoutAwareRefine:
    """Test timeout-aware prompt injection in iterative refine (R4-1b)."""

    def _prepare_timed_out_run(self, run_dir: Path) -> None:
        """Create a prior run that timed out with no metrics."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "task_id": "sandbox-main",
                "status": "failed",
                "metrics": {},
                "timed_out": True,
                "elapsed_sec": 120.0,
            }),
            encoding="utf-8",
        )
        # Write experiment code
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text(
            "print('best_loss: 0.1')\n",
            encoding="utf-8",
        )

    def test_timeout_refine_injects_scale_reduction_prompt(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        self._prepare_timed_out_run(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test topic",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 120,
                "max_iterations": 1,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient(
            "```python\nimport numpy as np\nprint('best_loss: 0.1')\n```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received the timeout-aware prompt
        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "TIMED OUT" in user_msg
        assert "120" in user_msg


# ── R4-2: Data Integrity Enforcement Tests ───────────────────────────


class TestDataIntegrityBlock:
    """Test paper draft blocked when no metrics exist (R4-2a)."""

    def test_paper_draft_blocked_with_no_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Write prior artifacts with NO metrics
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # No experiment_summary.json, no run files with metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "failed", "metrics": {}, "timed_out": True}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Ensure domain detection returns an empirical domain so the block triggers
        from researchclaw.pipeline.stage_impls import _paper_writing
        monkeypatch.setattr(
            _paper_writing, "_detect_domain",
            lambda topic, domains=(): ("ml", "machine learning", "NeurIPS, ICML, ICLR"),
        )

        llm = FakeLLMClient("should not be called")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        assert result.status == StageStatus.FAILED
        draft = (stage_dir / "paper_draft.md").read_text(encoding="utf-8")
        assert "Blocked" in draft or "BLOCKED" in draft or "no metrics" in draft.lower()
        # LLM should NOT have been called
        assert len(llm.calls) == 0

    def test_paper_draft_proceeds_with_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # Write experiment data with real metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"best_loss": 0.123},
                "stdout": "best_loss: 0.123\n",
            }),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nSome abstract text.")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        # Should proceed (LLM was called)
        assert len(llm.calls) >= 1
        # The prompt should contain anti-fabrication instructions
        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Data Integrity" in all_prompts or "ONLY report numbers" in all_prompts


# ── R4-3: Conference-Grade Title Guidelines Tests ────────────────────


class TestTitleGuidelines:
    """Test title_guidelines and abstract_structure blocks (R4-3)."""

    def test_title_guidelines_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("title_guidelines")
        assert "novelty" in block.lower() or "TITLE RULES" in block
        assert "14 words" in block or "15 words" in block or "concrete" in block.lower()

    def test_abstract_structure_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("abstract_structure")
        assert "5-sentence" in block or "problem" in block.lower()

    def test_title_guidelines_injected_into_paper_draft(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n")
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "completed",
                        "metrics": {"best_loss": 0.1}, "stdout": "best_loss: 0.1\n"}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nText.")
        rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Title" in all_prompts or "TITLE" in all_prompts
        assert 'Use EXACTLY this paper title: "Exact Configured Test Title"' in all_prompts


# ── R4-4: Conference-Grade Writing Quality Tests ─────────────────────


class TestConferenceWritingQuality:
    """Test enhanced writing prompts and writing_guide.py (R4-4)."""

    def test_writing_guide_format_all(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips()
        assert "Conference Writing Best Practices" in result
        assert "Title" in result
        assert "Common Rejections" in result

    def test_writing_guide_format_subset(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips(["title", "abstract"])
        assert "Title" in result
        assert "Abstract" in result
        assert "Common Rejections" not in result

    def test_paper_draft_system_includes_principles(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test",
            topic_constraint="test",
            exp_metrics_instruction="test",
            citation_instruction="test",
            outline="test",
        )
        # System prompt should mention key principles
        assert "NOVELTY" in sp.system or "novelty" in sp.system.lower()
        assert "fabricate" in sp.system.lower() or "real experimental" in sp.system.lower()


# ── R5-1 & R5-2: Bug Fixes Tests ────────────────────────────────────


class TestRefineTimeoutAndIterationCap:
    """Test R5-1 (no 120s cap) and R5-2 (iteration cap raised to 10)."""

    def test_refine_timeout_uses_full_budget(self) -> None:
        """R5-1: Refine sandbox should NOT cap at 120s."""
        import ast
        import inspect

        source = inspect.getsource(rc_executor._execute_iterative_refine)
        tree = ast.parse(source)
        source_text = inspect.getsource(rc_executor._execute_iterative_refine)
        # Should NOT contain min(..., 120)
        assert "min(config.experiment.time_budget_sec, 120)" not in source_text

    def test_iteration_cap_is_10(self) -> None:
        """R5-2: Max iterations should be capped at 10, not 3."""
        import inspect

        source = inspect.getsource(rc_executor._execute_iterative_refine)
        assert "min(requested_iterations, 10)" in source
        assert "min(requested_iterations, 3)" not in source

    def test_refine_respects_high_iteration_count(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """R5-2: Setting max_iterations=7 should actually allow 7 iterations."""
        # Write prior run artifacts
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "completed",
                        "metrics": {"best_loss": 0.5}}),
            encoding="utf-8",
        )
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('best_loss: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 300,
                "max_iterations": 7,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM always returns same code — will trigger no_improvement early stop
        llm = FakeLLMClient("```python\nprint('best_loss: 0.5')\n```")

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        log = json.loads((stage_dir / "refinement_log.json").read_text(encoding="utf-8"))
        # Should have been allowed more than 3 iterations (capped at 7)
        assert log["max_iterations_executed"] == 7
        # But may have stopped early due to no_improvement_for_2_iterations
        assert len(log["iterations"]) >= 2


# ── R5-3: NaN/Divergence Fast-Fail Tests ────────────────────────────


class TestNaNDivergenceDetection:
    """Test NaN/Inf filtering and divergence detection (R5-3)."""

    def test_parse_metrics_filters_nan(self) -> None:
        from researchclaw.experiment.sandbox import parse_metrics

        stdout = "best_loss: 0.5\nbad_metric: nan\ngood_metric: 1.23\n"
        metrics = parse_metrics(stdout)
        assert "best_loss" in metrics
        assert "good_metric" in metrics
        assert "bad_metric" not in metrics  # NaN should be filtered

    def test_parse_metrics_filters_inf(self) -> None:
        from researchclaw.experiment.sandbox import parse_metrics

        stdout = "metric_a: inf\nmetric_b: -inf\nmetric_c: 0.42\n"
        metrics = parse_metrics(stdout)
        assert "metric_c" in metrics
        assert "metric_a" not in metrics
        assert "metric_b" not in metrics

    def test_detect_nan_divergence_finds_nan(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("loss: nan\nstep 5 done", "")
        assert result is not None
        assert "NaN" in result or "nan" in result.lower()

    def test_detect_nan_divergence_finds_diverging_loss(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("best_loss: 999.5\n", "")
        assert result is not None
        assert "loss" in result.lower() or "999" in result

    def test_detect_nan_divergence_returns_none_for_clean(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("best_loss: 0.123\naccuracy: 0.95\n", "")
        assert result is None

    def test_runtime_issues_detects_diverging_loss(self) -> None:
        from types import SimpleNamespace

        fake_result = SimpleNamespace(
            metrics={"best_loss": 500.0},
            stdout="best_loss: 500.0\n",
            stderr="",
        )
        issues = rc_executor._detect_runtime_issues(fake_result)
        assert "DIVERGING" in issues or "diverging" in issues.lower()

    def test_compute_budget_includes_nan_guard(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "NaN" in block or "nan" in block.lower() or "divergence" in block.lower()


# ── R5-4: Experiment Harness Template Tests ──────────────────────────


class TestExperimentHarness:
    """Test the immutable experiment harness (R5-4)."""

    def test_harness_should_stop(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1)
        assert not h.should_stop()  # Just created, not at 80% yet
        import time
        time.sleep(0.9)
        assert h.should_stop()  # Should be past 80% of 1s

    def test_harness_report_metric(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("best_loss", 0.123)
        captured = capsys.readouterr()
        assert "best_loss: 0.123" in captured.out
        assert h._metrics["best_loss"] == 0.123

    def test_harness_rejects_nan(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("nan"))
        captured = capsys.readouterr()
        assert "bad" not in h._metrics
        assert "non-finite" in captured.err.lower() or "WARNING" in captured.err

    def test_harness_rejects_inf(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("inf"))
        assert "bad" not in h._metrics

    def test_harness_finalize(self, tmp_path: Path) -> None:
        import os
        from researchclaw.experiment.harness_template import ExperimentHarness

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            h = ExperimentHarness(time_budget=60)
            h.report_metric("accuracy", 0.95)
            h.report_metric("loss", 0.05)
            h.log_result({"condition": "A", "value": 1.0})
            h.finalize()

            results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
            assert results["metrics"]["accuracy"] == 0.95
            assert results["metrics"]["loss"] == 0.05
            assert len(results["results"]) == 1
        finally:
            os.chdir(old_cwd)

    def test_harness_progress(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1000)
        assert h.progress < 0.01  # Just started
        assert 0.0 <= h.progress <= 1.0

    def test_harness_injected_into_sandbox(self, tmp_path: Path) -> None:
        import sys
        from researchclaw.config import SandboxConfig
        from researchclaw.experiment.sandbox import ExperimentSandbox

        config = SandboxConfig(python_path=sys.executable)
        sandbox = ExperimentSandbox(config, tmp_path / "sandbox")

        # Create a project dir
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('test: 1.0')\n", encoding="utf-8")

        sandbox.run_project(project, timeout_sec=5)

        # Check that harness was injected (BUG-DA8-06: dir is now _project_{N})
        project_dirs = list((tmp_path / "sandbox").glob("_project_*"))
        assert project_dirs, "No _project_N directory found"
        harness_path = project_dirs[0] / "experiment_harness.py"
        assert harness_path.exists()
        content = harness_path.read_text(encoding="utf-8")
        assert "ExperimentHarness" in content

    def test_harness_not_overwritten_by_project(self, tmp_path: Path) -> None:
        import sys
        from researchclaw.config import SandboxConfig
        from researchclaw.experiment.sandbox import ExperimentSandbox

        config = SandboxConfig(python_path=sys.executable)
        sandbox = ExperimentSandbox(config, tmp_path / "sandbox")

        # Create a project with a fake experiment_harness.py
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('test: 1.0')\n", encoding="utf-8")
        (project / "experiment_harness.py").write_text("# FAKE HARNESS", encoding="utf-8")

        sandbox.run_project(project, timeout_sec=5)

        # The real harness should be there, not the fake one (BUG-DA8-06)
        project_dirs = list((tmp_path / "sandbox").glob("_project_*"))
        assert project_dirs
        harness_path = project_dirs[0] / "experiment_harness.py"
        content = harness_path.read_text(encoding="utf-8")
        assert "ExperimentHarness" in content
        assert "FAKE HARNESS" not in content

    def test_prompt_mentions_harness(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "experiment_harness" in block or "ExperimentHarness" in block


# ── R5-5: Stdout Truncation Tests ────────────────────────────────────


class TestStdoutTruncation:
    """Test stdout/stderr truncation in refine run summaries (R5-5)."""

    def test_long_stdout_truncated_in_refine(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        # Create a run with very long stdout
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        long_stdout = "\n".join(f"step {i}: loss={0.5 - i * 0.001:.6f}" for i in range(200))
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"best_loss": 0.3},
                "stdout": long_stdout,
            }),
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('best_loss: 0.3')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('best_loss: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received truncated stdout, not all 200 lines
        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        # Should contain truncation indicator
        assert "truncated" in user_msg or len(user_msg) < len(long_stdout)


# ===================================================================
# R6 Tests — Post-E2E Failure Analysis Fixes
# ===================================================================


class TestNoImproveStreakFix:
    """R6-1: no_improve_streak should only count iterations with real metrics."""

    def test_empty_metrics_dont_increment_streak(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """When metrics are empty (None), the streak should NOT increment."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "failed",
                "metrics": {},
                "stdout": "FAIL: NaN/divergence detected",
            }),
            encoding="utf-8",
        )
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 4,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM returns code that won't produce metrics in simulated mode
        llm = FakeLLMClient("```python\nprint('no metrics here')\n```")
        result = rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # Should abort after 3 consecutive no-metrics iterations
        log_path = stage_dir / "refinement_log.json"
        log_data = json.loads(log_path.read_text())
        # consecutive_no_metrics triggers early abort after 3 iterations
        assert len(log_data["iterations"]) == 3
        assert log_data.get("stop_reason") == "consecutive_no_metrics"


class TestStdoutFailureDetection:
    """R6-2: Detect stdout failure signals even when exit code is 0."""

    def test_fail_signal_in_stdout_marks_failed(self, tmp_path: Path) -> None:
        """Exit code 0 + 'FAIL:' in stdout + no metrics → status='failed'."""
        from researchclaw.pipeline.executor import _execute_experiment_run

        # Create necessary structure
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "stage-10").mkdir()
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir()
        # Simple code that prints FAIL but exits 0
        (exp_dir / "main.py").write_text(
            "print('FAIL: NaN/divergence detected')\n", encoding="utf-8"
        )
        (run_dir / "stage-11").mkdir()
        (run_dir / "stage-11" / "schedule.json").write_text("{}", encoding="utf-8")

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 512,
                    "allowed_imports": ["json"],
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        result = _execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        # Check the run payload
        runs_dir = stage_dir / "runs"
        run_file = runs_dir / "run-1.json"
        assert run_file.exists()
        payload = json.loads(run_file.read_text())
        assert payload["status"] == "failed"

    def test_clean_exit_no_fail_signal_marks_completed(self, tmp_path: Path) -> None:
        """Exit code 0 + valid metrics + no FAIL signal → status='completed'."""
        from researchclaw.pipeline.executor import _execute_experiment_run

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "stage-10").mkdir()
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir()
        (exp_dir / "main.py").write_text(
            "print('primary_metric: 0.95')\n", encoding="utf-8"
        )
        (run_dir / "stage-11").mkdir()
        (run_dir / "stage-11" / "schedule.json").write_text("{}", encoding="utf-8")

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 512,
                    "allowed_imports": ["json"],
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        result = _execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        runs_dir = stage_dir / "runs"
        payload = json.loads((runs_dir / "run-1.json").read_text())
        assert payload["status"] == "completed"


class TestMetricValUndefined:
    """R6-3: metric_val should be initialized to None before conditional block."""

    def test_metric_val_initialized_before_use(self) -> None:
        """Verify the code pattern: metric_val = None before if block."""
        import inspect
        source = inspect.getsource(rc_executor._execute_iterative_refine)
        # Find that metric_val = None appears before the sandbox block
        init_pos = source.find("metric_val = None")
        sandbox_pos = source.find("if validation.ok and config.experiment.mode")
        assert init_pos != -1, "metric_val = None not found"
        assert sandbox_pos != -1, "sandbox block not found"
        assert init_pos < sandbox_pos, "metric_val = None should come before sandbox block"


class TestConsecutiveEmptyMetrics:
    """R6-4: Pipeline should detect consecutive empty-metrics REFINE cycles."""

    def test_detects_consecutive_empty(self, tmp_path: Path) -> None:
        """Two cycles with empty metrics should return True."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        # Current cycle (stage-14)
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))
        # Previous cycle (stage-14_v1)
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is True

    def test_not_empty_when_metrics_exist(self, tmp_path: Path) -> None:
        """If any cycle has real metrics, return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {"loss": 0.5}},
        }))
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False

    def test_false_when_no_previous_cycle(self, tmp_path: Path) -> None:
        """First cycle (no v1) should return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        # No stage-14_v1 exists
        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False


# ===================================================================
# R7 Tests — Experiment-Paper Quality Alignment
# ===================================================================


class TestMultiConditionEnforcement:
    """R7-1: Code generation prompt must enforce multi-condition experiments."""

    def test_code_generation_prompt_has_multi_condition_block(self) -> None:
        """The code_generation prompt should contain multi-condition instructions."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test topic",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions:\n  - echo_chamber\n  - bridge_building\n  - random",
        )
        assert "MULTI-CONDITION REQUIREMENT" in sp.user
        assert "condition=" in sp.user
        assert "SUMMARY" in sp.user

    def test_multi_condition_labels_required(self) -> None:
        """Prompt must mention per-condition labeled output format."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="loss",
            pkg_hint="",
            exp_plan="treatments: [A, B, C]",
        )
        assert "condition=<name>" in sp.user


class TestEvidenceBoundedWriting:
    """R7-2: Paper draft prompt must enforce evidence-bounded claims."""

    def test_paper_draft_has_evidence_bounding_rules(self) -> None:
        """System prompt should contain evidence-bounding rules."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="# Outline",
        )
        assert "EVIDENCE-BOUNDING RULES" in sp.system
        assert "title" in sp.system.lower()
        assert "causal claim" in sp.system.lower() or "causal claims" in sp.system.lower()

    def test_hedging_language_guidance(self) -> None:
        """Should suggest hedged alternatives like 'Toward...' for partial data."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="",
        )
        assert "Toward" in sp.system or "Investigating" in sp.system


class TestConditionCoverageDetection:
    """R7-3: REFINE should detect condition coverage gaps."""

    def test_coverage_hint_injected_when_no_labels(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """If stdout has no 'condition=' labels, a coverage hint should be injected."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "primary_metric: 0.5\nprimary_metric: 0.3\n",
            }),
            encoding="utf-8",
        )

        exp_plan_dir = run_dir / "stage-09"
        exp_plan_dir.mkdir(parents=True, exist_ok=True)
        (exp_plan_dir / "exp_plan.yaml").write_text(
            "conditions:\n  - echo_chamber\n  - bridge_building\n  - random\n",
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('primary_metric: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "CONDITION COVERAGE GAP" in user_msg

    def test_no_hint_when_labels_present(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """If stdout already has 'condition=' labels, no hint should be injected."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "condition=echo primary_metric: 0.5\ncondition=bridge primary_metric: 0.3\n",
            }),
            encoding="utf-8",
        )

        exp_plan_dir = run_dir / "stage-09"
        exp_plan_dir.mkdir(parents=True, exist_ok=True)
        (exp_plan_dir / "exp_plan.yaml").write_text(
            "conditions:\n  - echo\n  - bridge\n",
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('primary_metric: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "CONDITION COVERAGE GAP" not in user_msg


# ===================================================================
# R8 Tests — AutoBench Round 1 Fixes
# ===================================================================


class TestBreadthFirstPrompt:
    """R8-1: Code generation prompt should require breadth-first condition ordering."""

    def test_breadth_first_in_code_generation(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [A, B, C]",
        )
        assert "BREADTH-FIRST" in sp.user
        assert "ONE representative" in sp.user


class TestRefineFilePreservation:
    """R8-2: Refine should preserve supporting files when LLM only returns main.py."""

    def test_supporting_files_preserved_in_refine(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """When LLM returns only main.py, other project files should be preserved."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "primary_metric: 0.5",
            }),
            encoding="utf-8",
        )

        # Multi-file experiment project
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("from helpers import foo\nprint('primary_metric: 0.5')\n")
        (exp_dir / "helpers.py").write_text("def foo(): return 42\n")
        (exp_dir / "utils.py").write_text("def bar(): return 99\n")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM returns only main.py in multi-file format
        llm = FakeLLMClient("```filename:main.py\nfrom helpers import foo\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # Check that experiment_v1 has ALL files, not just main.py
        v1_dir = stage_dir / "experiment_v1"
        assert v1_dir.exists()
        v1_files = {f.name for f in v1_dir.glob("*.py")}
        assert "main.py" in v1_files
        assert "helpers.py" in v1_files, "Supporting file helpers.py should be preserved"
        assert "utils.py" in v1_files, "Supporting file utils.py should be preserved"


# ===================================================================
# R9 Tests — AutoBench Round 2 Fixes
# ===================================================================


class TestCodeGenTopicNeutral:
    """R9-1: Code generation prompt should be topic-neutral, not optimization-biased."""

    def test_no_gradient_descent_bias(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should NOT contain optimization-specific examples as recommended approaches
        assert "Adam" not in sp.user
        assert "SGD" not in sp.user
        assert "Rosenbrock" not in sp.user
        # "gradient descent" may appear as anti-pattern warning but not as example
        assert "e.g., gradient descent" not in sp.user

    def test_topic_relevant_guidance(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should contain generic guidance that works for any topic
        assert "simulation" in sp.user.lower() or "appropriate" in sp.user.lower()
        assert "ACTUAL experiment" in sp.user or "relevant to the TOPIC" in sp.user


class TestRefineTopicAlignment:
    """R9-2: Refine prompt should include topic-code alignment check."""

    def test_topic_alignment_in_refine_prompt(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.sub_prompt(
            "iterative_improve",
            metric_key="primary_metric",
            metric_direction="maximize",
            files_context="# main.py\nprint('hello')",
            run_summaries="{}",
            condition_coverage_hint="",
            topic="multi-agent diversity scaling",
            exp_plan_anchor="",
        )
        assert "EXPERIMENT PLAN ANCHOR" in sp.user
        assert "multi-agent diversity scaling" in sp.user
        assert "NEVER rename" in sp.user


# =====================================================================
# _validate_draft_quality tests
# =====================================================================


def _make_prose(word_count: int) -> str:  # noqa: E302
    """Generate flowing prose text of approximately *word_count* words."""
    sentence = (
        "This is a flowing academic prose sentence "
        "that demonstrates our research findings. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_bullets(word_count: int) -> str:
    """Generate bullet-point text of approximately *word_count* words."""
    line = "- This is a bullet point about a research finding\n"
    words_per = len(line.split())
    return line * (word_count // words_per + 1)


def _make_comparative_prose(word_count: int) -> str:
    """Generate related-work style prose with comparative language."""
    sentence = (
        "Unlike prior work that focuses on simple baselines, "
        "our approach differs by incorporating novel techniques. "
        "In contrast to existing methods, we address key limitations. "
        "However, while previous approaches rely on heuristics, "
        "our method provides theoretical guarantees. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_results_prose(word_count: int) -> str:
    """Generate results prose with statistical measures."""
    sentence = (
        "Our method achieves 85.3 ± 1.2 accuracy averaged over 5 seeds. "
        "The baseline comparison yields a p-value of 0.003, confirming "
        "statistical significance with 95% confidence interval. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _build_draft(**section_overrides: str) -> str:
    """Build a paper draft with default prose sections."""
    defaults = {
        "Abstract": _make_prose(200),
        "Introduction": _make_prose(900),
        "Related Work": _make_comparative_prose(700),
        "Method": _make_prose(1200),
        "Experiments": _make_prose(1000),
        "Results": _make_results_prose(700),
        "Discussion": _make_prose(500),
        "Limitations": _make_prose(250),
        "Conclusion": _make_prose(250),
    }
    defaults.update(section_overrides)
    parts = ["# My Research Title\n"]
    for heading, body in defaults.items():
        parts.append(f"# {heading}\n{body}\n")
    return "\n".join(parts)


class TestValidateDraftQuality:
    """Tests for _validate_draft_quality()."""

    def test_short_section_triggers_warning(self) -> None:
        """Short Method section triggers expand warning."""
        draft = _build_draft(Method=_make_prose(200))
        result = rc_executor._validate_draft_quality(draft)
        assert any("Method" in w for w in result["overall_warnings"])
        assert any("EXPAND" in d or "Expand" in d
                    for d in result["revision_directives"])

    def test_bullet_density_triggers_warning(self) -> None:
        """Bullet-heavy Method section triggers rewrite warning."""
        draft = _build_draft(Method=_make_bullets(1200))
        result = rc_executor._validate_draft_quality(draft)
        assert any(
            "bullet" in w.lower() or "density" in w.lower()
            for w in result["overall_warnings"]
        )
        assert any("REWRITE" in d for d in result["revision_directives"])

    def test_clean_draft_no_warnings(self) -> None:
        """Balanced prose draft produces zero warnings."""
        draft = _build_draft()
        result = rc_executor._validate_draft_quality(draft)
        assert len(result["overall_warnings"]) == 0
        assert len(result["revision_directives"]) == 0

    def test_balance_warning(self) -> None:
        """Large imbalance between sections triggers balance warning."""
        draft = _build_draft(
            Introduction=_make_prose(1500),
            Results=_make_prose(100),
        )
        result = rc_executor._validate_draft_quality(draft)
        bal = [w for w in result["overall_warnings"]
               if "imbalance" in w.lower()]
        assert len(bal) >= 1, (
            f"Expected balance warning, got: {result['overall_warnings']}"
        )

    def test_writes_json_to_stage_dir(self, tmp_path: Path) -> None:
        """Quality report is written as draft_quality.json."""
        draft = _build_draft(Method=_make_prose(200))
        rc_executor._validate_draft_quality(draft, stage_dir=tmp_path)
        assert (tmp_path / "draft_quality.json").exists()
        data = json.loads(
            (tmp_path / "draft_quality.json").read_text(encoding="utf-8")
        )
        assert "section_analysis" in data
        assert "overall_warnings" in data
        assert "revision_directives" in data


class TestExperimentValidatorPrecision:
    def test_deep_validation_detects_undefined_helper_calls(self) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "def main():\n"
                    "    create_empty_csv('tmp.csv', ['a'])\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
            }
        )

        assert any(
            "Call to undefined function 'create_empty_csv()'" in issue
            for issue in issues
        )

    def test_deep_validation_allows_inherited_single_core_method_subclass(
        self,
    ) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "class BaseVerifier:\n"
                    "    def __init__(self, scale=1.0):\n"
                    "        self.scale = float(scale)\n\n"
                    "class ChildVerifier(BaseVerifier):\n"
                    "    def predict(self, value):\n"
                    "        total = value * self.scale\n"
                    "        shifted = total + 1.0\n"
                    "        centered = shifted - 0.5\n"
                    "        bounded = max(centered, 0.0)\n"
                    "        return {'score': bounded}\n"
                )
            }
        )

        assert not any(
            "Class 'ChildVerifier' has only 1 non-dunder method" in issue
            for issue in issues
        )

    def test_deep_validation_detects_duplicate_algorithm_classes_across_files(
        self,
    ) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "class DuplicateVerifier:\n"
                    "    def __init__(self, bias=0.0):\n"
                    "        self.bias = float(bias)\n\n"
                    "    def predict(self, value):\n"
                    "        shifted = value + self.bias\n"
                    "        bounded = max(shifted, 0.0)\n"
                    "        return {'score': bounded}\n"
                ),
                "models.py": (
                    "class DuplicateVerifier:\n"
                    "    def __init__(self, bias=0.0):\n"
                    "        self.bias = float(bias)\n\n"
                    "    def predict(self, value):\n"
                    "        shifted = value + self.bias\n"
                    "        bounded = max(shifted, 0.0)\n"
                    "        return {'score': bounded}\n"
                ),
            }
        )

        assert any(
            "Class 'DuplicateVerifier' is defined in multiple files" in issue
            for issue in issues
        )
        assert not any(
            "Classes 'DuplicateVerifier' and 'DuplicateVerifier' have identical"
            in issue
            for issue in issues
        )
