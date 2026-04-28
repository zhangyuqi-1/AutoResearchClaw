from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import cast
from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

from lxml import etree

logger = logging.getLogger(__name__)


_IMAGE_RE = re.compile(r"!\[[^\]]*]\((charts/[^)]+)\)")
_GENERIC_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_BOLD_FIGURE_CAPTION_RE = re.compile(
    r"^\*\*\s*Figure\s+\d+[.:]?\s*\*\*\s*(.*)\s*$",
    re.IGNORECASE,
)
_ITALIC_FIGURE_CAPTION_RE = re.compile(
    r"^\*\s*Figure\s+\d+[.:]?\s*(.*)\*\s*$",
    re.IGNORECASE,
)
_PLAIN_FIGURE_CAPTION_RE = re.compile(
    r"^Figure\s+\d+[.:]\s+.+$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_CAPTION_RE = re.compile(
    r"^(?:\*\*|\*)?\s*Table\s+\d+[.:]\s+.*(?:\*\*|\*)?\s*$",
    re.IGNORECASE,
)
_FIGURE_REFERENCE_CAPTURE_RE = re.compile(r"\bFigure\s+(\d+)\b", re.IGNORECASE)
_TABLE_REFERENCE_CAPTURE_RE = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
_GENERIC_ITALIC_BLOCK_RE = re.compile(r"^\*(?!\*)(.+?)\*\s*$", re.DOTALL)
_DOCX_CITATION_BLOCK_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9:_\-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9:_\-]*)+)\]")
_DOCX_LATEX_FIGURE_ENV_RE = re.compile(
    r"\\begin\{figure\}.*?\\end\{figure\}",
    re.DOTALL,
)
_DOCX_LATEX_TABLE_ENV_RE = re.compile(
    r"\\begin\{table\}.*?\\end\{table\}",
    re.DOTALL,
)
_DOCX_INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics(?:\[[^\]]*])?\{([^}]+)\}",
    re.IGNORECASE,
)
_DOCX_CAPTION_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
_DOCX_LABEL_RE = re.compile(r"\\label\{[^}]+\}")
_KEYWORDS_BLOCK_RE = re.compile(r"^\*\*Keywords:\*\*\s*(.+?)\s*$", re.IGNORECASE)
_DISPLAY_EQUATION_BLOCK_RE = re.compile(
    r"^\s*(?:\$\$.*\$\$|\\\[.*\\\]|\\begin\{equation\}.*\\end\{equation\})\s*$",
    re.DOTALL,
)
_DOCX_EQUATION_LAYOUT_CAPTION = "RCEquationLayout"
_EQUATION_REFERENCE_RE = re.compile(r"(?i)\bequation\s*\(\d+\)")
_EQUATION_EXPLANATION_PREFIX_RE = re.compile(r"(?i)^in\s+equation\s+\(\d+\),\s*")
_RAW_EQUATION_TOKEN_RE = re.compile(
    r"\\mathcal\{L\}(?:_\{[^}]+\}|_[A-Za-z0-9]+)?|\\sigma|\\mathrm\{[A-Za-z]+\}|[A-Za-z]+(?:_[A-Za-z0-9]+)?"
)
_DOCX_EQUATION_OPERATOR_ONLY_RE = re.compile(
    r"^(=|[+\-]|\\pm|\\mp|\\times|\\cdot|\\leq?|\\geq?|\\approx|\\sim|\\to|\\propto)\s*$"
)
_DOCX_EQUATION_CONTINUATION_RE = re.compile(
    r"^(=|[+\-]|\\pm|\\mp|\\times|\\cdot|\\leq?|\\geq?|\\approx|\\sim|\\to|\\propto)(?:\s+|$)"
)
_DOCX_EQUATION_RELATION_RE = re.compile(
    r"(=|\\leq?|\\geq?|\\approx|\\sim|\\to|\\propto)"
)
_GENERIC_EQUATION_SYMBOLS = {
    "arg",
    "bar",
    "ge",
    "in",
    "mathcal",
    "max",
    "min",
    "operatorname",
    "quad",
    "star",
    "tilde",
}
_GENERIC_EQUATION_EXPLANATION_RE = re.compile(
    r"\bdenotes a variable defined in the surrounding text\b",
    re.IGNORECASE,
)
_BAD_EQUATION_EXPLANATION_PHRASES = (
    "denotes the probe function",
)

_KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "approach",
    "based",
    "for",
    "from",
    "in",
    "of",
    "on",
    "paper",
    "study",
    "system",
    "systems",
    "the",
    "to",
    "toward",
    "using",
    "via",
    "with",
}
_EQUATION_TOKEN_STOPWORDS = {
    "begin",
    "end",
    "equation",
    "frac",
    "left",
    "log",
    "right",
    "sum",
    "text",
    "top",
}
_DOCX_COMPRESSION_PRIORITY = (
    "Conclusion",
    "Discussion",
    "Limitations",
    "Related Work",
    "Introduction",
    "Results and Analysis",
    "Results",
    "Experimental Setup",
    "Experiments",
    "Method",
)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_DOCX_NS = {"w": _W_NS, "m": _M_NS}


@dataclass
class _Bundle:
    image_path: str
    alt_text: str
    image_index: int
    start: int
    end: int
    caption_index: int | None
    explanation_indices: tuple[int, ...]
    section: str
    figure_number: int | None


@dataclass
class _CodexLoopResult:
    success: bool
    markdown: str
    review: dict[str, object]
    iterations: list[dict[str, object]]
    assessment: dict[str, object]
    error: str = ""


_CITATION_BRACKET_RE = re.compile(r"\[([A-Za-z0-9_,;\- ]+)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+\.\d+|\d+)(?:%?)")


def _split_blocks(markdown: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", markdown.strip())]
    return [block for block in blocks if block]


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(blocks).strip() + "\n"


def _section_contexts(blocks: list[str]) -> list[str]:
    current = ""
    contexts: list[str] = []
    for block in blocks:
        match = _HEADING_RE.match(block)
        if match and len(match.group(1)) <= 2:
            current = match.group(2).strip()
        contexts.append(current)
    return contexts


def _is_caption_block(block: str) -> bool:
    stripped = block.strip()
    return bool(
        _BOLD_FIGURE_CAPTION_RE.match(stripped)
        or _ITALIC_FIGURE_CAPTION_RE.match(stripped)
        or _PLAIN_FIGURE_CAPTION_RE.match(stripped)
    )


def _docx_reference_doc_path() -> Path:
    return Path(__file__).resolve().parents[2] / "templates" / "styles" / "reference.docx"


def _split_markdown_sections(markdown: str) -> list[tuple[int, str, str]]:
    matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.*?)\s*$", markdown))
    if not matches:
        return []
    sections: list[tuple[int, str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        sections.append((len(match.group(1)), match.group(2).strip(), markdown[start:end].strip()))
    return sections


def _render_markdown_sections(
    title_block: tuple[int, str, str] | None,
    sections: list[tuple[int, str, str]],
) -> str:
    parts: list[str] = []
    if title_block is not None:
        level, heading, body = title_block
        parts.append("#" * level + f" {heading}")
        if body.strip():
            parts.extend(["", body.strip()])
    for level, heading, body in sections:
        if parts:
            parts.append("")
        parts.append("#" * level + f" {heading}")
        if body.strip():
            parts.extend(["", body.strip()])
    return "\n".join(parts).strip() + "\n"


def _strip_fenced_code_blocks(markdown: str) -> str:
    return re.sub(
        r"(?ms)^```[^\n]*\n.*?^```[ \t]*\n?",
        "\n",
        markdown,
    )


def _markdown_integrity_issues(
    markdown: str,
    *,
    baseline: str | None = None,
) -> list[str]:
    issues: list[str] = []
    fence_markers = len(re.findall(r"(?m)^```", markdown))
    if fence_markers % 2 != 0:
        issues.append("unbalanced_code_fences")
    stripped = _strip_fenced_code_blocks(markdown)
    sections = _split_markdown_sections(stripped)
    if not sections:
        issues.append("missing_heading_structure")
        return issues

    headings = [heading.strip().lower() for _, heading, _ in sections]
    if baseline is not None:
        baseline_sections = _split_markdown_sections(_strip_fenced_code_blocks(baseline))
        baseline_headings = [heading.strip().lower() for _, heading, _ in baseline_sections]
        if baseline_headings and headings and headings[0] != baseline_headings[0]:
            issues.append("title_changed_or_missing")
        canonical_required = {
            heading
            for heading in baseline_headings
            if heading in {
                "abstract",
                "introduction",
                "method",
                "experiments",
                "results",
                "results and analysis",
                "conclusion",
            }
        }
        missing = canonical_required - set(headings)
        if missing:
            issues.append("missing_required_headings:" + ",".join(sorted(missing)))
    return issues


def _extract_image_alt_text(block: str) -> str:
    match = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", block)
    if not match:
        return ""
    return match.group(1).strip()


def _keyword_overlap(text: str, reference: str) -> int:
    stopwords = {
        "about",
        "across",
        "adult",
        "analysis",
        "chart",
        "comparison",
        "covtype",
        "dataset",
        "datasets",
        "figure",
        "heatmap",
        "main",
        "method",
        "methods",
        "results",
        "showing",
        "shows",
        "the",
        "this",
        "what",
    }
    source_tokens = {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token not in stopwords
    }
    reference_tokens = {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", reference.lower())
        if token not in stopwords
    }
    return len(source_tokens & reference_tokens)


def _extract_topic_keywords(topic: str, domains: tuple[str, ...], *, limit: int = 5) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", topic):
        lowered = token.lower()
        if lowered in _KEYWORD_STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(lowered)
        if len(keywords) >= limit:
            return keywords
    for domain in domains:
        lowered = str(domain).strip().lower()
        if (
            not lowered
            or lowered in _KEYWORD_STOPWORDS
            or lowered in seen
        ):
            continue
        seen.add(lowered)
        keywords.append(lowered)
        if len(keywords) >= limit:
            break
    return keywords


def _append_block(body: str, block: str) -> str:
    body = body.strip()
    block = block.strip()
    if not block:
        return body
    if not body:
        return block
    return body + "\n\n" + block


def _keyword_block_score(keywords_text: str) -> tuple[int, int]:
    terms = [term.strip() for term in keywords_text.split(",") if term.strip()]
    return (len(terms), len(keywords_text.strip()))


def _select_best_keywords(candidates: list[str]) -> str:
    if not candidates:
        return ""
    return max(candidates, key=_keyword_block_score).strip()


def _deduplicate_keyword_blocks_in_body(body: str) -> tuple[str, str]:
    blocks = _split_blocks(body)
    if not blocks:
        return body.strip(), ""
    kept: list[str] = []
    candidates: list[str] = []
    for block in blocks:
        match = _KEYWORDS_BLOCK_RE.match(block.strip())
        if match:
            candidates.append(match.group(1).strip())
            continue
        kept.append(block.strip())
    return _join_blocks(kept), _select_best_keywords(candidates)


def _deduplicate_keywords_in_sections(
    sections: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    selected_keywords = ""
    cleaned_sections: list[tuple[int, str, str]] = []
    for level, heading, body in sections:
        cleaned_body, keywords = _deduplicate_keyword_blocks_in_body(body)
        if keywords and (
            not selected_keywords
            or _keyword_block_score(keywords) > _keyword_block_score(selected_keywords)
        ):
            selected_keywords = keywords
        cleaned_sections.append((level, heading, cleaned_body))

    if not selected_keywords:
        return cleaned_sections

    keyword_block = "**Keywords:** " + selected_keywords
    for idx, (level, heading, body) in enumerate(cleaned_sections):
        if heading.strip().lower() == "abstract":
            cleaned_sections[idx] = (level, heading, _append_block(body, keyword_block))
            return cleaned_sections
    if cleaned_sections:
        level, heading, body = cleaned_sections[0]
        cleaned_sections[0] = (level, heading, _append_block(body, keyword_block))
    return cleaned_sections


def _sections_have_keywords(sections: list[tuple[int, str, str]]) -> bool:
    return any(
        _KEYWORDS_BLOCK_RE.match(block.strip())
        for _, _, body in sections
        for block in _split_blocks(body)
    )


def _merge_section_into_body(body: str, label: str, extra_body: str) -> str:
    extra_body = extra_body.strip()
    if not extra_body:
        return body.strip()
    return _append_block(body, f"**{label}:**\n{extra_body}")


def _table_block_signature(block: str) -> str:
    return "\n".join(
        re.sub(r"\s+", " ", line.strip())
        for line in block.splitlines()
        if line.strip()
    )


def _extract_figure_number(text: str) -> int | None:
    match = _FIGURE_REFERENCE_CAPTURE_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_table_number(text: str) -> int | None:
    match = _TABLE_REFERENCE_CAPTURE_RE.search(text)
    return int(match.group(1)) if match else None


def _clean_table_caption_text(block: str) -> str:
    text = block.strip()
    text = re.sub(r"^(?:\*\*|\*)?\s*Table\s+\d+[.:]\s*", "", text, flags=re.IGNORECASE)
    return text.strip("* ").strip()


def _strip_sentence_end(text: str) -> str:
    return text.strip().rstrip(".,;: ")


def _lower_sentence_lead(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return cleaned.lower()
    if cleaned[0].isalpha() and cleaned[1].islower():
        return cleaned[0].lower() + cleaned[1:]
    return cleaned


def _is_reference_candidate_block(block: str) -> bool:
    stripped = block.strip()
    return bool(
        stripped
        and not _HEADING_RE.match(stripped)
        and not _IMAGE_RE.search(stripped)
        and not _is_caption_block(stripped)
        and not _TABLE_CAPTION_RE.match(stripped)
        and not stripped.startswith("|")
        and not stripped.startswith(">")
    )


def _find_nearby_numbered_reference_block(
    blocks: list[str],
    *,
    start: int,
    end: int,
    pattern: re.Pattern[str],
) -> str | None:
    candidate_indices: list[int] = []
    for offset in (1, 2):
        before_idx = start - offset
        if before_idx >= 0:
            candidate_indices.append(before_idx)
    for offset in (1, 2):
        after_idx = end + offset
        if after_idx < len(blocks):
            candidate_indices.append(after_idx)
    for idx in candidate_indices:
        block = blocks[idx].strip()
        if _is_reference_candidate_block(block) and pattern.search(block):
            return block
    return None


def _build_figure_caption_map(sources: tuple[str, ...]) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    for source in sources:
        source_blocks = _split_blocks(source)
        for bundle in _extract_bundles(source_blocks):
            if bundle.caption_index is None:
                continue
            key = Path(bundle.image_path).name
            caption = source_blocks[bundle.caption_index].strip()
            if key and caption and key not in caption_map:
                caption_map[key] = caption
    return caption_map


def _build_figure_reference_map(sources: tuple[str, ...]) -> dict[str, str]:
    reference_map: dict[str, str] = {}
    for source in sources:
        source_blocks = _split_blocks(source)
        for bundle in _extract_bundles(source_blocks):
            if bundle.figure_number is None:
                continue
            key = Path(bundle.image_path).name
            if not key or key in reference_map:
                continue
            pattern = re.compile(rf"\bFigure\s+{bundle.figure_number}\b", re.IGNORECASE)
            reference = None
            for explanation_idx in bundle.explanation_indices:
                if 0 <= explanation_idx < len(source_blocks):
                    candidate = source_blocks[explanation_idx].strip()
                    if _is_reference_candidate_block(candidate) and pattern.search(candidate):
                        reference = candidate
                        break
            reference = reference or _find_nearby_numbered_reference_block(
                source_blocks,
                start=bundle.start,
                end=bundle.end,
                pattern=pattern,
            )
            if reference:
                reference_map[key] = reference.strip()
    return reference_map


def _build_table_caption_map(sources: tuple[str, ...]) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    for source in sources:
        source_blocks = _split_blocks(source)
        for idx, block in enumerate(source_blocks):
            stripped = block.strip()
            if not stripped.startswith("|"):
                continue
            if idx == 0:
                continue
            caption = source_blocks[idx - 1].strip()
            if not _TABLE_CAPTION_RE.match(caption):
                continue
            signature = _table_block_signature(stripped)
            if signature and signature not in caption_map:
                caption_map[signature] = caption
    return caption_map


def _build_table_reference_map(sources: tuple[str, ...]) -> dict[str, str]:
    reference_map: dict[str, str] = {}
    for source in sources:
        source_blocks = _split_blocks(source)
        for idx, block in enumerate(source_blocks):
            stripped = block.strip()
            if not stripped.startswith("|") or idx == 0:
                continue
            caption = source_blocks[idx - 1].strip()
            if not _TABLE_CAPTION_RE.match(caption):
                continue
            table_number = _extract_table_number(caption)
            signature = _table_block_signature(stripped)
            if table_number is None or not signature or signature in reference_map:
                continue
            pattern = re.compile(rf"\bTable\s+{table_number}\b", re.IGNORECASE)
            reference = _find_nearby_numbered_reference_block(
                source_blocks,
                start=idx - 1,
                end=idx,
                pattern=pattern,
            )
            if reference:
                reference_map[signature] = reference.strip()
    return reference_map


def _restore_table_captions_in_body(body: str, caption_map: dict[str, str]) -> str:
    blocks = _split_blocks(body)
    if not blocks or not caption_map:
        return body.strip()
    restored: list[str] = []
    for block in blocks:
        stripped = block.strip()
        if stripped.startswith("|"):
            signature = _table_block_signature(stripped)
            caption = caption_map.get(signature, "")
            if caption and not (restored and _TABLE_CAPTION_RE.match(restored[-1].strip())):
                restored.append(caption)
        restored.append(stripped)
    return _join_blocks(restored)


def _restore_figure_captions_in_body(body: str, caption_map: dict[str, str]) -> str:
    blocks = _split_blocks(body)
    if not blocks or not caption_map:
        return body.strip()
    restored: list[str] = []
    for idx, block in enumerate(blocks):
        stripped = block.strip()
        restored.append(stripped)
        image_match = _IMAGE_RE.search(stripped)
        if not image_match:
            continue
        has_prev_caption = len(restored) >= 2 and _is_caption_block(restored[-2].strip())
        has_next_caption = idx + 1 < len(blocks) and _is_caption_block(blocks[idx + 1].strip())
        if has_prev_caption or has_next_caption:
            continue
        caption = caption_map.get(Path(image_match.group(1)).name, "").strip()
        if caption:
            restored.append(caption)
    return _join_blocks(restored)


def _figure_reference_fallback(figure_number: int, caption_block: str, alt_text: str) -> str:
    caption_text = _clean_caption_text(caption_block)
    if not caption_text:
        caption_text = alt_text.strip()
    caption_text = _strip_sentence_end(caption_text)
    if not caption_text:
        return f"Figure {figure_number} summarizes the local visual evidence."
    return f"Figure {figure_number} summarizes {_lower_sentence_lead(caption_text)}."


def _table_reference_fallback(table_number: int, caption_block: str) -> str:
    caption_text = _strip_sentence_end(_clean_table_caption_text(caption_block))
    if not caption_text:
        return f"Table {table_number} reports the local benchmark values."
    return f"Table {table_number} reports {_lower_sentence_lead(caption_text)}."


def _restore_figure_references_in_body(
    body: str,
    reference_map: dict[str, str],
) -> str:
    blocks = _split_blocks(body)
    if not blocks:
        return body.strip()
    for bundle in reversed(_extract_bundles(blocks)):
        if bundle.figure_number is None:
            continue
        if _find_first_explicit_figure_reference_index(blocks, bundle.figure_number) is not None:
            continue
        image_name = Path(bundle.image_path).name
        caption_block = (
            blocks[bundle.caption_index]
            if bundle.caption_index is not None and 0 <= bundle.caption_index < len(blocks)
            else ""
        )
        reference = reference_map.get(image_name, "").strip()
        if not reference:
            reference = _figure_reference_fallback(
                bundle.figure_number,
                caption_block,
                bundle.alt_text,
            )
        replaced = False
        for explanation_idx in bundle.explanation_indices:
            if 0 <= explanation_idx < len(blocks) and not _FIGURE_REFERENCE_CAPTURE_RE.search(
                blocks[explanation_idx]
            ):
                blocks[explanation_idx] = reference
                replaced = True
                break
        if not replaced:
            blocks = blocks[: bundle.start] + [reference] + blocks[bundle.start :]
    return _join_blocks(blocks)


def _restore_table_references_in_body(
    body: str,
    reference_map: dict[str, str],
) -> str:
    blocks = _split_blocks(body)
    if not blocks:
        return body.strip()
    idx = len(blocks) - 1
    while idx >= 0:
        block = blocks[idx].strip()
        if not block.startswith("|") or idx == 0:
            idx -= 1
            continue
        caption = blocks[idx - 1].strip()
        if not _TABLE_CAPTION_RE.match(caption):
            idx -= 1
            continue
        table_number = _extract_table_number(caption)
        if table_number is None:
            idx -= 1
            continue
        if _find_first_explicit_table_reference_index(blocks, table_number) is not None:
            idx -= 1
            continue
        signature = _table_block_signature(block)
        reference = reference_map.get(signature, "").strip()
        if not reference:
            reference = _table_reference_fallback(table_number, caption)
        insert_at = idx - 1
        generic_idx = insert_at - 1
        if generic_idx >= 0 and _is_reference_candidate_block(blocks[generic_idx]):
            lowered = blocks[generic_idx].lower()
            if "below" in lowered or "summarized" in lowered or "summarised" in lowered:
                blocks[generic_idx] = reference
                idx -= 1
                continue
        blocks = blocks[:insert_at] + [reference] + blocks[insert_at:]
        idx -= 1
    return _join_blocks(blocks)


def _is_equation_explanation_block(block: str) -> bool:
    lowered = block.strip().lower()
    return (
        lowered.startswith("where ")
        or lowered.startswith("here,")
        or lowered.startswith("in this expression,")
        or lowered.startswith("in this equation,")
        or bool(_EQUATION_EXPLANATION_PREFIX_RE.match(block.strip()))
    )


def _looks_like_display_equation(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if _DISPLAY_EQUATION_BLOCK_RE.match(stripped):
        return True
    if any(
        pattern.match(stripped)
        for pattern in (
            _HEADING_RE,
            _TABLE_CAPTION_RE,
            _BOLD_FIGURE_CAPTION_RE,
            _ITALIC_FIGURE_CAPTION_RE,
        )
    ):
        return False
    if _GENERIC_IMAGE_RE.search(stripped) or "|" in stripped:
        return False
    if "=" not in stripped:
        return False
    if len(stripped.split()) > 18:
        return False
    return bool(re.search(r"[\\^_{}]|[α-ωΑ-ΩσΣℒ]", stripped))


def _normalize_equation_lead_in(block: str) -> str:
    stripped = block.strip()
    stripped = re.sub(
        r"\s+(?:as\s+shown\s+|as\s+defined\s+)?in\s+Equation\s+\(\d+\)(?=[:.]|\s*$)",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    stripped = re.sub(
        r"(?i)\bEquation\s+\(\d+\)\s+(?:defines|shows|gives)\s+",
        "",
        stripped,
        count=1,
    ).strip()
    if stripped.endswith((",", ";")):
        return stripped[:-1] + ":"
    if stripped.endswith(":"):
        return stripped
    if stripped.endswith("."):
        return stripped
    return stripped + ":"


def _unwrap_display_equation(block: str) -> str:
    stripped = block.strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        inner = stripped[2:-2].strip()
    elif stripped.startswith("\\[") and stripped.endswith("\\]"):
        inner = stripped[2:-2].strip()
    else:
        inner = stripped
    inner = re.sub(r"[ \t]*[.,;:]+$", "", inner)
    return inner.strip()


def _normalize_display_equation(block: str) -> str:
    inner = _unwrap_display_equation(block)
    return "$$\n" + inner + "\n$$"


def _find_top_level_relation(line: str) -> tuple[int, str] | None:
    commands = ("\\leq", "\\geq", "\\approx", "\\sim", "\\to", "\\propto")
    depth = 0
    idx = 0
    while idx < len(line):
        for command in commands:
            if depth == 0 and line.startswith(command, idx):
                return idx, command
        char = line[idx]
        if char == "\\":
            idx += 1
            while idx < len(line) and line[idx].isalpha():
                idx += 1
            continue
        if char == "{":
            depth += 1
            idx += 1
            continue
        if char == "}":
            depth = max(0, depth - 1)
            idx += 1
            continue
        if depth == 0 and char == "=":
            return idx, char
        idx += 1
    return None


def _split_docx_equation_relation(line: str) -> tuple[str, str, str] | None:
    stripped = line.strip()
    match = _find_top_level_relation(stripped)
    if match is None:
        return None
    start, relation = match
    left = stripped[:start].rstrip()
    right = stripped[start + len(relation) :].lstrip()
    if not left or not right:
        return None
    return left, relation, right


def _split_docx_equation_leading_relation(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    for relation in ("\\leq", "\\geq", "\\approx", "\\sim", "\\to", "\\propto", "="):
        if stripped.startswith(relation):
            right = stripped[len(relation) :].lstrip()
            if right:
                return relation, right
    return None


def _collapse_docx_equation_lines(lines: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(line.strip() for line in lines if line.strip())).strip()


def _equation_symbol_key(symbol: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", symbol.lower().strip("$").lstrip("\\"))


def _docx_equation_should_align(lines: list[str]) -> bool:
    if len(lines) <= 1:
        return False
    continuation_lines = 0
    relation_lines = 0
    for line in lines[1:]:
        if _split_docx_equation_leading_relation(line) is not None:
            continuation_lines += 1
            continue
        if _DOCX_EQUATION_CONTINUATION_RE.match(line.strip()):
            continuation_lines += 1
    for line in lines:
        if _split_docx_equation_relation(line) is not None:
            relation_lines += 1
    collapsed = _collapse_docx_equation_lines(lines)
    return continuation_lines >= 2 or (relation_lines > 1 and len(collapsed) > 120)


def _normalize_docx_display_equation(block: str) -> str:
    inner = _unwrap_display_equation(block)
    if (
        not inner
        or "\\begin{aligned}" in inner
        or "\\begin{array}" in inner
        or "\\begin{split}" in inner
        or "\\\\" in inner
    ):
        return _normalize_display_equation(block)

    lines = [line.strip() for line in inner.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _normalize_display_equation(block)

    merged: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if idx + 1 < len(lines) and _DOCX_EQUATION_OPERATOR_ONLY_RE.fullmatch(line):
            merged.append(f"{line} {lines[idx + 1].lstrip()}".strip())
            idx += 2
            continue
        merged.append(line)
        idx += 1

    if len(merged) <= 1:
        return _normalize_display_equation(block)

    if not _docx_equation_should_align(merged):
        return "$$\n" + _collapse_docx_equation_lines(merged) + "\n$$"

    rows: list[str] = []
    start_idx = 1
    first_relation = _split_docx_equation_relation(merged[0])
    if first_relation is not None:
        left, relation, right = first_relation
        rows.append(f"{left} &{relation} {right}")
    elif len(merged) > 1 and _split_docx_equation_leading_relation(merged[1]) is not None:
        relation, right = _split_docx_equation_leading_relation(merged[1]) or ("", "")
        rows.append(f"{merged[0]} &{relation} {right}".rstrip())
        start_idx = 2
    elif len(merged) > 1 and _DOCX_EQUATION_CONTINUATION_RE.match(merged[1]):
        rows.append(f"{merged[0]} &{merged[1]}")
        start_idx = 2
    else:
        rows.append(merged[0])

    for line in merged[start_idx:]:
        if not line:
            continue
        leading_relation = _split_docx_equation_leading_relation(line)
        if leading_relation is not None:
            relation, right = leading_relation
            rows.append(f"&{relation} {right}")
            continue
        relation = _split_docx_equation_relation(line)
        if relation is not None:
            left, rel, right = relation
            rows.append(f"{left} &{rel} {right}")
            continue
        if _DOCX_EQUATION_CONTINUATION_RE.match(line):
            rows.append(f"&\\quad {line}")
            continue
        rows.append(f"&\\quad {line}")

    aligned = " \\\\\n".join(rows)
    return "$$\n\\begin{aligned}\n" + aligned + "\n\\end{aligned}\n$$"


def _extract_equation_symbols(equation_body: str) -> list[str]:
    normalized = equation_body.replace("\\mathrm{LN}", "LN")
    symbols: list[str] = []
    seen: set[str] = set()
    for token in _RAW_EQUATION_TOKEN_RE.findall(normalized):
        cleaned = token.strip()
        if not cleaned:
            continue
        lower = cleaned.lower().lstrip("\\")
        if (
            lower in _EQUATION_TOKEN_STOPWORDS
            or lower.startswith("sum")
            or lower.startswith("log")
        ):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        symbols.append(cleaned)
    return symbols


def _format_equation_symbol(symbol: str) -> str:
    if symbol == "LN":
        return r"$\mathrm{LN}$"
    return f"${symbol}$"


def _describe_equation_symbol(symbol: str, lead_context: str) -> str:
    lowered_context = lead_context.lower()
    plain = symbol.lower().lstrip("\\")
    if plain in _GENERIC_EQUATION_SYMBOLS:
        return ""
    if symbol == "\\sigma":
        return "denotes the sigmoid activation"
    if symbol == "LN":
        return "denotes layer normalization"
    if plain.startswith("mathcal{l}") or plain.startswith("l_"):
        if "loss" in lowered_context or "cross-entropy" in lowered_context:
            return "denotes the training loss"
        return "denotes the objective defined in this section"
    if plain.startswith("y"):
        if "probability" in lowered_context or "classifier" in lowered_context:
            return "denotes the predicted probability"
        return "denotes the model output"
    if plain.startswith("p"):
        if "probe" in lowered_context:
            return "denotes the probe prediction"
        return "denotes the predicted probability"
    if plain.startswith("q"):
        return "denotes the probe function"
    if plain.startswith(("w", "w_")) or symbol.startswith(("W", "w")):
        return "denotes a learnable weight parameter"
    if plain.startswith("b"):
        return "denotes a learnable bias parameter"
    if plain.startswith(("z", "x", "h")):
        return "denotes the representation for sample $i$"
    if plain.startswith("t") and "threshold" in lowered_context:
        return "denotes the validation threshold"
    return ""


def _join_explanation_fragments(fragments: list[str]) -> str:
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]
    if len(fragments) == 2:
        return fragments[0] + ", and " + fragments[1]
    return ", ".join(fragments[:-1]) + ", and " + fragments[-1]


def _build_equation_explanation(equation_block: str, lead_context: str) -> str:
    symbols = _extract_equation_symbols(_unwrap_display_equation(equation_block))
    if not symbols:
        return ""
    fragments = []
    for symbol in symbols[:7]:
        description = _describe_equation_symbol(symbol, lead_context)
        if not description:
            continue
        fragments.append(f"{_format_equation_symbol(symbol)} {description}")
    if not fragments:
        return ""
    return "Here, " + _join_explanation_fragments(fragments) + "."


def _ensure_equation_reference(block: str, equation_number: int) -> str:
    reference = f"Equation ({equation_number})"
    stripped = block.strip()
    if not stripped:
        return stripped
    if _EQUATION_REFERENCE_RE.search(stripped):
        return _EQUATION_REFERENCE_RE.sub(reference, stripped)
    if stripped.endswith(":"):
        return stripped[:-1].rstrip() + f" in {reference}:"
    if stripped.endswith("."):
        return stripped[:-1].rstrip() + f" in {reference}."
    return stripped + f" in {reference}:"


def _normalize_equation_explanation_block(block: str, equation_number: int) -> str:
    _ = equation_number
    stripped = block.strip()
    replacements = (
        (r"(?i)^in\s+equation\s+\(\d+\),\s*", ""),
        (r"(?i)^here,\s*", ""),
        (r"(?i)^in\s+this\s+equation,\s*", ""),
        (r"(?i)^in\s+this\s+expression,\s*", ""),
        (r"(?i)^where\s+", ""),
    )
    body = stripped
    for pattern, replacement in replacements:
        body = re.sub(pattern, replacement, body, count=1)
    body = body.strip()
    fragments = [frag.strip() for frag in re.split(r",\s*(?=\$)", body) if frag.strip()]
    kept_fragments: list[str] = []
    for fragment in fragments or ([body] if body else []):
        cleaned_fragment = re.sub(r"^(?:and\s+)", "", fragment).strip()
        if not cleaned_fragment:
            continue
        if _GENERIC_EQUATION_EXPLANATION_RE.search(cleaned_fragment):
            continue
        matched_bad_phrase = next(
            (
                phrase
                for phrase in _BAD_EQUATION_EXPLANATION_PHRASES
                if phrase in cleaned_fragment.lower()
            ),
            "",
        )
        if matched_bad_phrase:
            symbol_match_for_phrase = re.match(r"^\$([^$]+)\$", cleaned_fragment)
            symbol_key_for_phrase = (
                _equation_symbol_key(symbol_match_for_phrase.group(1))
                if symbol_match_for_phrase
                else ""
            )
            if symbol_key_for_phrase in _GENERIC_EQUATION_SYMBOLS or matched_bad_phrase == "denotes the probe function":
                continue
        symbol_match = re.match(r"^\$([^$]+)\$", cleaned_fragment)
        if symbol_match and _equation_symbol_key(symbol_match.group(1)) in _GENERIC_EQUATION_SYMBOLS:
            continue
        kept_fragments.append(cleaned_fragment)
    body = _join_explanation_fragments(kept_fragments).strip()
    body = re.sub(r"\s+,", ",", body)
    body = re.sub(r",\s*,", ", ", body)
    body = re.sub(r"\s+", " ", body).strip(" ,")
    if _GENERIC_EQUATION_EXPLANATION_RE.search(body) or not re.search(r"[A-Za-z0-9$\\]", body):
        body = ""
    if not body:
        return ""
    return f"Here, {body}"


def _normalize_section_equations_numbered(
    body: str,
    *,
    start_number: int = 1,
) -> tuple[str, int]:
    blocks = _split_blocks(body)
    if not blocks:
        return body.strip(), start_number

    normalized: list[str] = []
    equation_number = start_number
    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        stripped = block.strip()
        if not _looks_like_display_equation(stripped):
            normalized.append(stripped)
            idx += 1
            continue

        lead_context = ""
        if normalized and not _looks_like_display_equation(normalized[-1]):
            normalized[-1] = _normalize_equation_lead_in(normalized[-1])
            lead_context = normalized[-1]

        normalized_equation = _normalize_display_equation(stripped)
        normalized.append(normalized_equation)
        next_block = blocks[idx + 1].strip() if idx + 1 < len(blocks) else ""
        if _is_equation_explanation_block(next_block):
            explanation = _normalize_equation_explanation_block(next_block, equation_number)
            if explanation:
                normalized.append(explanation)
            idx += 2
        else:
            explanation = _build_equation_explanation(normalized_equation, lead_context)
            if explanation:
                normalized_explanation = _normalize_equation_explanation_block(
                    explanation,
                    equation_number,
                )
                if normalized_explanation:
                    normalized.append(normalized_explanation)
            idx += 1
        equation_number += 1

    return _join_blocks(normalized), equation_number


def _normalize_section_equations(body: str) -> str:
    normalized, _ = _normalize_section_equations_numbered(body, start_number=1)
    return normalized


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w\-]+\b", text))


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _truncate_prose_block(block: str, word_budget: int) -> str:
    if word_budget <= 0:
        return ""
    kept: list[str] = []
    used = 0
    for sentence in _split_sentences(block):
        sentence_words = _count_words(sentence)
        if kept and used + sentence_words > word_budget:
            break
        kept.append(sentence)
        used += sentence_words
        if used >= word_budget:
            break
    return " ".join(kept).strip()


def _section_word_budget(
    heading: str,
    overflow_pages: int,
    submission_profile: str,
    compression_round: int,
) -> int:
    heading_lower = heading.strip().lower()
    base_budgets = {
        "conclusion": 160,
        "discussion": 260,
        "limitations": 180,
        "related work": 420 if submission_profile != "ei_conference" else 220,
        "introduction": 520,
        "results": 620,
        "results and analysis": 760,
        "experimental setup": 520,
        "experiments": 560,
        "method": 780,
    }
    reductions = {
        "conclusion": 40,
        "discussion": 60,
        "limitations": 50,
        "related work": 90,
        "introduction": 50,
        "results": 50,
        "results and analysis": 60,
        "experimental setup": 40,
        "experiments": 40,
        "method": 35,
    }
    base = base_budgets.get(heading_lower, 500)
    reduction = reductions.get(heading_lower, 30) * max(overflow_pages, 1) * max(compression_round, 1)
    return max(80, base - reduction)


def _compress_section_body(
    body: str,
    *,
    heading: str,
    overflow_pages: int,
    submission_profile: str,
    compression_round: int,
) -> str:
    budget = _section_word_budget(
        heading,
        overflow_pages,
        submission_profile,
        compression_round,
    )
    blocks = _split_blocks(body)
    if not blocks:
        return body.strip()

    compressed: list[str] = []
    used_words = 0
    for block in blocks:
        stripped = block.strip()
        if (
            _looks_like_display_equation(stripped)
            or _is_equation_explanation_block(stripped)
            or _GENERIC_IMAGE_RE.search(stripped)
            or _TABLE_CAPTION_RE.match(stripped)
            or _is_caption_block(stripped)
            or stripped.startswith("|")
        ):
            compressed.append(stripped)
            continue
        remaining = budget - used_words
        if remaining <= 0:
            continue
        truncated = _truncate_prose_block(stripped, remaining)
        if not truncated:
            continue
        compressed.append(truncated)
        used_words += _count_words(truncated)
    return _join_blocks(compressed) if compressed else body.strip()


def _compress_markdown_for_docx_limit(
    markdown: str,
    *,
    current_page_count: int,
    page_limit: int,
    submission_profile: str,
    compression_round: int = 1,
) -> str:
    overflow_pages = current_page_count - page_limit
    if overflow_pages <= 0:
        return markdown

    sections = _split_markdown_sections(markdown)
    if not sections:
        return markdown

    title_block: tuple[int, str, str] | None = None
    body_sections = sections
    if sections[0][0] == 1:
        title_block = sections[0]
        body_sections = sections[1:]

    priority_rank = {
        heading.lower(): idx for idx, heading in enumerate(_DOCX_COMPRESSION_PRIORITY)
    }
    updated_sections = [(level, heading, body) for level, heading, body in body_sections]
    for idx, (level, heading, body) in sorted(
        enumerate(updated_sections),
        key=lambda item: priority_rank.get(item[1][1].lower(), len(priority_rank)),
    ):
        compressed_body = _compress_section_body(
            body,
            heading=heading,
            overflow_pages=overflow_pages,
            submission_profile=submission_profile,
            compression_round=compression_round,
        )
        updated_sections[idx] = (level, heading, compressed_body)

    return _render_markdown_sections(title_block, updated_sections)


def _extend_unique(target: list[str], items: list[str]) -> None:
    for item in items:
        if item not in target:
            target.append(item)


def _load_docx_quality_payload(docx_quality_path: Path) -> dict[str, object]:
    if docx_quality_path.exists():
        return cast(
            dict[str, object],
            json.loads(docx_quality_path.read_text(encoding="utf-8")),
        )
    return {
        "clean": False,
        "heading_numbering_ok": False,
        "equation_alignment_ok": False,
        "display_math_omml_ok": False,
        "figure_caption_numbering_ok": False,
        "table_caption_numbering_ok": False,
        "issues": ["docx_not_exported"],
    }


def _normalize_final_paper_markdown(
    markdown: str,
    *,
    topic: str,
    domains: tuple[str, ...],
    submission_profile: str,
    table_caption_sources: tuple[str, ...] = (),
    figure_reference_sources: tuple[str, ...] = (),
) -> str:
    sections = _split_markdown_sections(markdown)
    if not sections:
        return markdown.strip() + ("\n" if markdown.strip() else "")

    title_block: tuple[int, str, str] | None = None
    body_sections = sections
    if sections[0][0] == 1:
        title_block = sections[0]
        body_sections = sections[1:]

    normalized_sections: list[tuple[int, str, str]] = []
    equation_number = 1
    for level, heading, body in body_sections:
        normalized_body, equation_number = _normalize_section_equations_numbered(
            body.strip(),
            start_number=equation_number,
        )
        normalized_sections.append((level, heading, normalized_body))
    caption_map = _build_table_caption_map(table_caption_sources)
    if caption_map:
        normalized_sections = [
            (level, heading, _restore_table_captions_in_body(body, caption_map))
            for level, heading, body in normalized_sections
        ]
    figure_sources = figure_reference_sources or table_caption_sources
    figure_caption_map = _build_figure_caption_map(figure_sources)
    if figure_caption_map:
        normalized_sections = [
            (level, heading, _restore_figure_captions_in_body(body, figure_caption_map))
            for level, heading, body in normalized_sections
        ]
    figure_reference_map = _build_figure_reference_map(figure_sources)
    if figure_reference_map or figure_caption_map:
        normalized_sections = [
            (level, heading, _restore_figure_references_in_body(body, figure_reference_map))
            for level, heading, body in normalized_sections
        ]
    table_reference_map = _build_table_reference_map(table_caption_sources)
    if table_reference_map or caption_map:
        normalized_sections = [
            (level, heading, _restore_table_references_in_body(body, table_reference_map))
            for level, heading, body in normalized_sections
        ]
    normalized_sections = _deduplicate_keywords_in_sections(normalized_sections)
    if not _sections_have_keywords(normalized_sections):
        keywords = _extract_topic_keywords(topic, domains)
        if keywords:
            keyword_block = "**Keywords:** " + ", ".join(keywords)
            for idx, (level, heading, body) in enumerate(normalized_sections):
                if heading.strip().lower() == "abstract":
                    normalized_sections[idx] = (
                        level,
                        heading,
                        _append_block(body, keyword_block),
                    )
                    break

    if submission_profile != "ei_conference":
        return _render_markdown_sections(title_block, normalized_sections)

    intro_idx: int | None = None
    conclusion_idx: int | None = None
    related_body = ""
    discussion_body = ""
    limitations_body = ""
    profiled_sections: list[tuple[int, str, str]] = []

    for level, heading, body in normalized_sections:
        lowered = heading.strip().lower()
        if lowered == "introduction" and intro_idx is None:
            intro_idx = len(profiled_sections)
        if lowered == "conclusion" and conclusion_idx is None:
            conclusion_idx = len(profiled_sections)
        if lowered == "related work":
            related_body = _append_block(related_body, body)
            continue
        if lowered == "results":
            heading = "Results and Analysis"
        elif lowered == "discussion":
            discussion_body = _append_block(discussion_body, body)
            continue
        elif lowered == "limitations":
            limitations_body = _append_block(limitations_body, body)
            continue
        profiled_sections.append((level, heading, body))

    if related_body:
        if intro_idx is not None:
            level, heading, body = profiled_sections[intro_idx]
            profiled_sections[intro_idx] = (
                level,
                heading,
                _merge_section_into_body(body, "Related work synthesis", related_body),
            )
        elif profiled_sections:
            level, heading, body = profiled_sections[0]
            profiled_sections[0] = (
                level,
                heading,
                _merge_section_into_body(body, "Related work synthesis", related_body),
            )

    merged_conclusion = _merge_section_into_body("", "Discussion", discussion_body)
    merged_conclusion = _merge_section_into_body(merged_conclusion, "Limitations", limitations_body)
    if merged_conclusion:
        if conclusion_idx is not None:
            level, heading, body = profiled_sections[conclusion_idx]
            profiled_sections[conclusion_idx] = (
                level,
                heading,
                _append_block(body, merged_conclusion),
            )
        else:
            profiled_sections.append((2, "Conclusion", merged_conclusion))

    return _render_markdown_sections(title_block, profiled_sections)


def _yaml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _rewrite_docx_citations(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        parts = [part.strip() for part in raw.split(",")]
        if not parts or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9:_\-]*", part) for part in parts):
            return match.group(0)
        return "[" + "; ".join(f"@{part}" for part in parts) + "]"

    return _DOCX_CITATION_BLOCK_RE.sub(_replace, text)


def _extract_docx_citation_clusters(text: str) -> list[list[str]]:
    pattern = re.compile(r"\[([A-Za-z][A-Za-z0-9:_\-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9:_\-]*)*)\]")
    clusters: list[list[str]] = []
    for match in pattern.finditer(text):
        raw = match.group(1)
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts and all(_looks_like_citation_key(part) for part in parts):
            clusters.append(parts)
    return clusters


def _collect_citation_keys(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for cluster in _extract_docx_citation_clusters(text):
        for key in cluster:
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _enforce_reference_limit(markdown: str, *, max_references: int) -> str:
    if max_references <= 0:
        return markdown

    allowed_order: list[str] = []
    allowed_set: set[str] = set()
    pattern = re.compile(r"(?<!\!)\[([^\[\]]+)\]")

    def _replace(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if not parts or not all(_looks_like_citation_key(part) for part in parts):
            return match.group(0)
        kept: list[str] = []
        for part in parts:
            if part in allowed_set:
                kept.append(part)
                continue
            if len(allowed_order) >= max_references:
                continue
            allowed_order.append(part)
            allowed_set.add(part)
            kept.append(part)
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    limited = pattern.sub(_replace, markdown)
    limited = re.sub(r"[ \t]+\n", "\n", limited)
    limited = re.sub(r"\n{3,}", "\n\n", limited)
    limited = re.sub(r"\s+([,.;:])", r"\1", limited)
    return limited


def _preserves_required_structure(original_markdown: str, candidate_markdown: str) -> bool:
    return not _markdown_integrity_issues(candidate_markdown, baseline=original_markdown)


def _sync_bibliography_with_markdown(stage_dir: Path, markdown: str) -> bool:
    bib_path = stage_dir / "references.bib"
    if not bib_path.exists():
        return False

    try:
        from researchclaw.pipeline.stage_impls._review_publish import (
            _dedupe_bibtex_entries,
            _extract_citation_keys_from_text,
            _remove_bibtex_entries,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 24: bibliography sync unavailable: %s", exc)
        return False

    try:
        bib_text = bib_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Stage 24: failed reading bibliography: %s", exc)
        return False

    deduped = _dedupe_bibtex_entries(bib_text)
    cited_keys = set(_collect_citation_keys(markdown)) | _extract_citation_keys_from_text(markdown)
    if cited_keys:
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", deduped))
        drop_keys = bib_keys - cited_keys
        if drop_keys:
            deduped = _remove_bibtex_entries(deduped, drop_keys)
            deduped = _dedupe_bibtex_entries(deduped)
    if deduped != bib_text:
        bib_path.write_text(deduped, encoding="utf-8")
    return True


def _format_docx_numeric_citations(text: str) -> tuple[str, list[str]]:
    clusters = _extract_docx_citation_clusters(text)
    ordered_keys: list[str] = []
    key_numbers: dict[str, int] = {}
    for cluster in clusters:
        for key in cluster:
            if key not in key_numbers:
                key_numbers[key] = len(ordered_keys) + 1
                ordered_keys.append(key)

    pattern = re.compile(r"\[([A-Za-z][A-Za-z0-9:_\-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9:_\-]*)*)\]")

    def _replace(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if not parts or not all(part in key_numbers for part in parts):
            return match.group(0)
        nums = [str(key_numbers[part]) for part in parts]
        return "[" + ", ".join(nums) + "]"

    return pattern.sub(_replace, text), ordered_keys


def _docx_caption_text(block: str) -> str:
    stripped = block.strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        stripped = stripped[2:-2].strip()
    elif stripped.startswith("*") and stripped.endswith("*"):
        stripped = stripped[1:-1].strip()
    return " ".join(stripped.split()).strip()


def _is_generic_italic_caption_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if _is_caption_block(stripped) or _TABLE_CAPTION_RE.match(stripped):
        return False
    if _HEADING_RE.match(stripped):
        return False
    if "\n" in stripped:
        return False
    match = _GENERIC_ITALIC_BLOCK_RE.match(stripped)
    return bool(match and match.group(1).strip())


def _normalize_docx_image_caption_text(block: str, figure_number: int) -> str:
    text = _docx_caption_text(block)
    if re.match(r"^Figure\s+\d+[.:]?\s+", text, re.IGNORECASE):
        return text
    return f"Figure {figure_number}. {text}"


def _format_docx_author_list(author_field: str) -> str:
    authors = [part.strip() for part in author_field.replace("\n", " ").split(" and ") if part.strip()]
    formatted: list[str] = []
    for author in authors:
        if "," in author:
            last, first = [part.strip() for part in author.split(",", 1)]
            formatted.append(f"{last}, {first}".strip().strip(","))
        else:
            formatted.append(author)
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def _format_docx_bibliography_entry(entry: dict[str, str]) -> str:
    parts: list[str] = []
    authors = _format_docx_author_list(entry.get("author", ""))
    if authors:
        parts.append(authors + ".")
    year = entry.get("year", "").strip()
    title = entry.get("title", "").strip().strip("{}")
    if year:
        parts.append(f"{year}.")
    if title:
        parts.append(f'"{title}."')
    venue = (
        entry.get("journal", "").strip()
        or entry.get("booktitle", "").strip()
        or entry.get("publisher", "").strip()
        or entry.get("howpublished", "").strip()
    )
    if venue:
        parts.append(venue + ".")
    doi = entry.get("doi", "").strip()
    url = entry.get("url", "").strip()
    if doi:
        parts.append(f"https://doi.org/{doi}.")
    elif url:
        parts.append(url + ".")
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _build_docx_references_section(bibliography_text: str, ordered_keys: list[str]) -> list[str]:
    if not bibliography_text.strip() or not ordered_keys:
        return []
    by_key: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}", bibliography_text, re.DOTALL):
        _entrytype, entry_id, body = match.groups()
        fields: dict[str, str] = {"ID": entry_id.strip()}
        for field_match in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?:,|$)", body, re.DOTALL):
            key, value = field_match.groups()
            fields[key.strip().lower()] = " ".join(value.strip().split())
        by_key[fields["ID"]] = fields
    lines = ["# References", ""]
    for idx, key in enumerate(ordered_keys, start=1):
        entry = by_key.get(key)
        formatted = _format_docx_bibliography_entry(entry) if entry else f"{key}."
        lines.extend(
            [
                '::: {custom-style="Bibliography"}',
                f"[{idx}] {formatted}",
                ":::",
                "",
            ]
        )
    return lines


def _normalize_docx_caption_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.replace(r"\_", "_")
    return cleaned.strip()


def _convert_docx_latex_figure_env(match: re.Match[str]) -> str:
    block = match.group(0)
    image_match = _DOCX_INCLUDEGRAPHICS_RE.search(block)
    caption_match = _DOCX_CAPTION_RE.search(block)
    image_path = image_match.group(1).strip() if image_match else ""
    caption = _normalize_docx_caption_text(caption_match.group(1)) if caption_match else ""
    parts: list[str] = []
    if image_path:
        parts.append(f"![{caption or 'Figure'}]({image_path})")
    if caption:
        parts.append(f"*{caption}*")
    return "\n\n".join(parts).strip()


def _convert_docx_latex_table_env(match: re.Match[str]) -> str:
    block = match.group(0)
    caption_match = _DOCX_CAPTION_RE.search(block)
    caption = _normalize_docx_caption_text(caption_match.group(1)) if caption_match else ""
    cleaned = _DOCX_CAPTION_RE.sub("", block)
    cleaned = _DOCX_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"\\begin\{table\}(\[[^\]]*])?", "", cleaned)
    cleaned = re.sub(r"\\end\{table\}", "", cleaned)
    cleaned = re.sub(r"\\centering", "", cleaned)
    cleaned = cleaned.strip()
    parts: list[str] = []
    if caption:
        parts.append(f"**{caption}**")
    if cleaned and "|" in cleaned:
        parts.append(cleaned)
    return "\n\n".join(parts).strip()


def _strip_latex_docx_blocks(text: str) -> str:
    text = _DOCX_LATEX_FIGURE_ENV_RE.sub(_convert_docx_latex_figure_env, text)
    text = _DOCX_LATEX_TABLE_ENV_RE.sub(_convert_docx_latex_table_env, text)
    return text


def _style_docx_caption_block(block: str) -> str:
    stripped = block.strip()
    style = "TableCaption" if _TABLE_CAPTION_RE.match(stripped) else "ImageCaption"
    cleaned = stripped
    if cleaned.startswith("**") and cleaned.endswith("**"):
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("*") and cleaned.endswith("*"):
        cleaned = cleaned[1:-1].strip()
    return f'::: {{custom-style="{style}"}}\n{cleaned}\n:::'


def _remove_image_alt_text(block: str) -> str:
    return _GENERIC_IMAGE_RE.sub(lambda m: f"![]({m.group(1)})", block)


def _normalize_docx_list_block(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    bullet_re = re.compile(r"^(\s*)[-*]\s+(.*)$")
    normalized: list[str] = []
    changed = False
    for line in lines:
        match = bullet_re.match(line)
        if not match:
            return block
        indent, body = match.groups()
        normalized.append(f"{indent}• {body.strip()}")
        changed = True
    return "\n".join(normalized) if changed else block


def _prepare_docx_markdown(
    markdown: str,
    *,
    authors: str,
    bibliography_name: str,
    bibliography_text: str = "",
    rewrite_citations: bool = True,
) -> str:
    markdown = _strip_latex_docx_blocks(markdown)
    _, ordered_keys = _format_docx_numeric_citations(markdown)
    sections = _split_markdown_sections(markdown)
    title = "Final Paper"
    shift = 0
    body_sections = sections
    if sections and sections[0][0] == 1:
        title = sections[0][1]
        body_sections = sections[1:]
        shift = 1
    body_sections = _deduplicate_keywords_in_sections(
        [(level, heading, body.strip()) for level, heading, body in body_sections]
    )

    abstract_body = ""
    normalized_sections: list[tuple[int, str, str]] = []
    for level, heading, body in body_sections:
        if heading.lower() == "abstract" and not abstract_body:
            abstract_body = body.strip()
            continue
        normalized_sections.append((max(1, level - shift), heading, body.strip()))

    parts = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f'author: "{_yaml_escape(authors)}"',
        'date: ""',
        "---",
        "",
    ]

    if abstract_body:
        parts.extend(
            [
                '::: {custom-style="Abstract"}',
                "Abstract",
                ":::",
                "",
            ]
        )
        abstract_text, _ = _format_docx_numeric_citations(abstract_body)
        abstract_blocks = _split_blocks(abstract_text)
        for idx, block in enumerate(abstract_blocks):
            style = "Keywords" if _KEYWORDS_BLOCK_RE.match(block.strip()) else ("FirstParagraph" if idx == 0 else "BodyText")
            if style == "Keywords":
                block = "Keywords: " + _KEYWORDS_BLOCK_RE.match(block.strip()).group(1)  # type: ignore[union-attr]
            parts.extend(
                [
                    f'::: {{custom-style="{style}"}}',
                    block.strip(),
                    ":::",
                    "",
                ]
            )

    figure_counter = 0
    for level, heading, body in normalized_sections:
        parts.append("#" * level + f" {heading}")
        parts.append("")
        body_text, _ = _format_docx_numeric_citations(body)
        body_blocks = _split_blocks(body_text)
        idx = 0
        while idx < len(body_blocks):
            block = body_blocks[idx].strip()
            next_block = body_blocks[idx + 1].strip() if idx + 1 < len(body_blocks) else ""
            if _IMAGE_RE.search(block):
                has_caption = bool(
                    next_block
                    and (
                        _is_caption_block(next_block)
                        or _TABLE_CAPTION_RE.match(next_block)
                        or _is_generic_italic_caption_block(next_block)
                    )
                )
                if has_caption:
                    block = _remove_image_alt_text(block)
                parts.append(block)
                parts.append("")
                if has_caption:
                    figure_counter += 1
                    if _TABLE_CAPTION_RE.match(next_block):
                        parts.append(_style_docx_caption_block(next_block))
                    else:
                        caption_text = _normalize_docx_image_caption_text(next_block, figure_counter)
                        parts.append(
                            _style_docx_caption_block(f"*{caption_text}*")
                        )
                    parts.append("")
                    idx += 2
                    continue
                idx += 1
                continue
            if _is_caption_block(block) or _TABLE_CAPTION_RE.match(block):
                parts.append(_style_docx_caption_block(block))
            elif _looks_like_display_equation(block):
                parts.append(_normalize_docx_display_equation(block))
            else:
                parts.append(_normalize_docx_list_block(block))
            parts.append("")
            idx += 1
    parts.extend(_build_docx_references_section(bibliography_text, ordered_keys))
    return "\n".join(parts).strip() + "\n"


def _w(name: str) -> str:
    return f"{{{_W_NS}}}{name}"


def _docx_paragraph_style(paragraph: etree._Element) -> str | None:
    style = paragraph.find("./w:pPr/w:pStyle", namespaces=_DOCX_NS)
    if style is None:
        return None
    return style.get(_w("val"))


def _docx_paragraph_text(paragraph: etree._Element) -> str:
    texts = paragraph.xpath(".//w:t/text()", namespaces=_DOCX_NS)
    return "".join(texts).strip()


def _docx_paragraph_has_payload(paragraph: etree._Element) -> bool:
    if _docx_paragraph_text(paragraph):
        return True
    return bool(
        paragraph.xpath(
            ".//w:drawing | .//m:oMath | .//m:oMathPara | .//w:tbl",
            namespaces=_DOCX_NS,
        )
    )


def _ensure_docx_heading_numbering(numbering_root: etree._Element) -> str:
    abstract_id = "4242"
    num_id = "4242"
    for child in list(numbering_root):
        if child.tag == _w("abstractNum") and child.get(_w("abstractNumId")) == abstract_id:
            numbering_root.remove(child)
        if child.tag == _w("num") and child.get(_w("numId")) == num_id:
            numbering_root.remove(child)

    abstract = etree.SubElement(numbering_root, _w("abstractNum"))
    abstract.set(_w("abstractNumId"), abstract_id)
    etree.SubElement(abstract, _w("multiLevelType")).set(_w("val"), "multilevel")
    patterns = ["%1.", "%1.%2", "%1.%2.%3"]
    for level, pattern in enumerate(patterns):
        lvl = etree.SubElement(abstract, _w("lvl"))
        lvl.set(_w("ilvl"), str(level))
        etree.SubElement(lvl, _w("start")).set(_w("val"), "1")
        etree.SubElement(lvl, _w("numFmt")).set(_w("val"), "decimal")
        etree.SubElement(lvl, _w("lvlText")).set(_w("val"), pattern)
        etree.SubElement(lvl, _w("lvlJc")).set(_w("val"), "left")
        ppr = etree.SubElement(lvl, _w("pPr"))
        etree.SubElement(ppr, _w("ind")).set(_w("left"), str(360 * (level + 1)))
    num = etree.SubElement(numbering_root, _w("num"))
    num.set(_w("numId"), num_id)
    etree.SubElement(num, _w("abstractNumId")).set(_w("val"), abstract_id)
    return num_id


def _ensure_docx_paragraph_style(styles_root: etree._Element, style_id: str) -> etree._Element:
    for style in styles_root.findall("./w:style", namespaces=_DOCX_NS):
        if style.get(_w("styleId")) == style_id:
            return style
    style = etree.SubElement(styles_root, _w("style"))
    style.set(_w("type"), "paragraph")
    style.set(_w("styleId"), style_id)
    etree.SubElement(style, _w("name")).set(_w("val"), style_id)
    return style


def _ensure_docx_style_spacing(style: etree._Element, *, before: int, after: int, line: int | None = None) -> None:
    ppr = style.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.SubElement(style, _w("pPr"))
    spacing = ppr.find("./w:spacing", namespaces=_DOCX_NS)
    if spacing is None:
        spacing = etree.SubElement(ppr, _w("spacing"))
    spacing.set(_w("before"), str(before))
    spacing.set(_w("after"), str(after))
    if line is not None:
        spacing.set(_w("line"), str(line))
        spacing.set(_w("lineRule"), "auto")


def _ensure_docx_style_indent(style: etree._Element, *, left: int = 0, first_line: int | None = None, hanging: int | None = None) -> None:
    ppr = style.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.SubElement(style, _w("pPr"))
    ind = ppr.find("./w:ind", namespaces=_DOCX_NS)
    if ind is None:
        ind = etree.SubElement(ppr, _w("ind"))
    ind.set(_w("left"), str(left))
    if first_line is not None:
        ind.set(_w("firstLine"), str(first_line))
    if hanging is not None:
        ind.set(_w("hanging"), str(hanging))


def _ensure_docx_style_keep(style: etree._Element, *, keep_next: bool = False, keep_lines: bool = False) -> None:
    ppr = style.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.SubElement(style, _w("pPr"))
    for tag, enabled in (("keepNext", keep_next), ("keepLines", keep_lines)):
        node = ppr.find(f"./w:{tag}", namespaces=_DOCX_NS)
        if enabled and node is None:
            etree.SubElement(ppr, _w(tag))
        if not enabled and node is not None:
            ppr.remove(node)


def _ensure_docx_style_justification(style: etree._Element, align: str) -> None:
    ppr = style.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.SubElement(style, _w("pPr"))
    jc = ppr.find("./w:jc", namespaces=_DOCX_NS)
    if jc is None:
        jc = etree.SubElement(ppr, _w("jc"))
    jc.set(_w("val"), align)


def _ensure_docx_style_run(
    style: etree._Element,
    *,
    bold: bool = False,
    italic: bool = False,
    size: int | None = None,
    color: str | None = None,
) -> None:
    rpr = style.find("./w:rPr", namespaces=_DOCX_NS)
    if rpr is None:
        rpr = etree.SubElement(style, _w("rPr"))
    for tag, enabled in (("b", bold), ("i", italic)):
        node = rpr.find(f"./w:{tag}", namespaces=_DOCX_NS)
        if enabled and node is None:
            etree.SubElement(rpr, _w(tag))
        if not enabled and node is not None:
            rpr.remove(node)
    if size is not None:
        sz = rpr.find("./w:sz", namespaces=_DOCX_NS)
        if sz is None:
            sz = etree.SubElement(rpr, _w("sz"))
        sz.set(_w("val"), str(size))
        szcs = rpr.find("./w:szCs", namespaces=_DOCX_NS)
        if szcs is None:
            szcs = etree.SubElement(rpr, _w("szCs"))
        szcs.set(_w("val"), str(size))
    color_node = rpr.find("./w:color", namespaces=_DOCX_NS)
    if color is None:
        if color_node is not None:
            rpr.remove(color_node)
    else:
        if color_node is None:
            color_node = etree.SubElement(rpr, _w("color"))
        color_node.set(_w("val"), color)


def _normalize_docx_styles(styles_root: etree._Element) -> None:
    title = _ensure_docx_paragraph_style(styles_root, "Title")
    _ensure_docx_style_spacing(title, before=180, after=90)
    _ensure_docx_style_keep(title, keep_next=True, keep_lines=True)
    _ensure_docx_style_justification(title, "center")
    _ensure_docx_style_run(title, bold=True, size=30, color="000000")

    author = _ensure_docx_paragraph_style(styles_root, "Author")
    _ensure_docx_style_spacing(author, before=0, after=60)
    _ensure_docx_style_keep(author, keep_next=True, keep_lines=True)
    _ensure_docx_style_justification(author, "center")
    _ensure_docx_style_run(author, size=18, color="000000")

    abstract = _ensure_docx_paragraph_style(styles_root, "Abstract")
    _ensure_docx_style_spacing(abstract, before=120, after=60)
    _ensure_docx_style_keep(abstract, keep_next=True, keep_lines=True)
    _ensure_docx_style_justification(abstract, "left")
    _ensure_docx_style_run(abstract, bold=True, size=24, color="000000")

    body = _ensure_docx_paragraph_style(styles_root, "BodyText")
    _ensure_docx_style_spacing(body, before=0, after=36, line=260)
    _ensure_docx_style_indent(body, left=0, first_line=360)
    _ensure_docx_style_justification(body, "both")
    _ensure_docx_style_run(body, size=24, color="000000")

    first = _ensure_docx_paragraph_style(styles_root, "FirstParagraph")
    _ensure_docx_style_spacing(first, before=0, after=36, line=260)
    _ensure_docx_style_indent(first, left=0, first_line=0)
    _ensure_docx_style_justification(first, "both")
    _ensure_docx_style_run(first, size=24, color="000000")

    for style_id, size, before, after in (
        ("Heading1", 30, 150, 48),
        ("Heading2", 28, 120, 36),
        ("Heading3", 28, 90, 24),
    ):
        heading = _ensure_docx_paragraph_style(styles_root, style_id)
        _ensure_docx_style_spacing(heading, before=before, after=after)
        _ensure_docx_style_keep(heading, keep_next=True, keep_lines=True)
        _ensure_docx_style_justification(heading, "left")
        _ensure_docx_style_run(heading, bold=True, size=size, color="000000")

    image_caption = _ensure_docx_paragraph_style(styles_root, "ImageCaption")
    _ensure_docx_style_spacing(image_caption, before=24, after=72)
    _ensure_docx_style_keep(image_caption, keep_next=True, keep_lines=True)
    _ensure_docx_style_justification(image_caption, "center")
    _ensure_docx_style_run(image_caption, italic=True, size=21, color="000000")

    table_caption = _ensure_docx_paragraph_style(styles_root, "TableCaption")
    _ensure_docx_style_spacing(table_caption, before=72, after=18)
    _ensure_docx_style_keep(table_caption, keep_next=True, keep_lines=True)
    _ensure_docx_style_justification(table_caption, "left")
    _ensure_docx_style_run(table_caption, bold=True, size=21, color="000000")

    bibliography = _ensure_docx_paragraph_style(styles_root, "Bibliography")
    _ensure_docx_style_spacing(bibliography, before=0, after=120, line=280)
    _ensure_docx_style_indent(bibliography, left=360, hanging=360)
    _ensure_docx_style_justification(bibliography, "left")
    _ensure_docx_style_run(bibliography, size=21, color="000000")

    keywords = _ensure_docx_paragraph_style(styles_root, "Keywords")
    _ensure_docx_style_spacing(keywords, before=0, after=60, line=240)
    _ensure_docx_style_indent(keywords, left=0, first_line=0)
    _ensure_docx_style_justification(keywords, "left")
    _ensure_docx_style_run(keywords, bold=True, size=21, color="000000")

    compact = _ensure_docx_paragraph_style(styles_root, "Compact")
    _ensure_docx_style_spacing(compact, before=0, after=0, line=240)
    _ensure_docx_style_indent(compact, left=0, first_line=0)
    _ensure_docx_style_justification(compact, "left")
    _ensure_docx_style_run(compact, size=21, color="000000")


def _apply_docx_page_layout(document_root: etree._Element) -> None:
    body = document_root.find("./w:body", namespaces=_DOCX_NS)
    if body is None:
        return
    sect = body.find("./w:sectPr", namespaces=_DOCX_NS)
    if sect is None:
        sect = etree.SubElement(body, _w("sectPr"))
    pg_sz = sect.find("./w:pgSz", namespaces=_DOCX_NS)
    if pg_sz is None:
        pg_sz = etree.SubElement(sect, _w("pgSz"))
    pg_sz.set(_w("w"), "12240")
    pg_sz.set(_w("h"), "15840")
    pg_mar = sect.find("./w:pgMar", namespaces=_DOCX_NS)
    if pg_mar is None:
        pg_mar = etree.SubElement(sect, _w("pgMar"))
    pg_mar.set(_w("top"), "1440")
    pg_mar.set(_w("right"), "1260")
    pg_mar.set(_w("bottom"), "1440")
    pg_mar.set(_w("left"), "1260")
    pg_mar.set(_w("header"), "720")
    pg_mar.set(_w("footer"), "720")
    pg_mar.set(_w("gutter"), "0")


def _apply_docx_heading_numbering(document_root: etree._Element, num_id: str) -> bool:
    applied = False
    for paragraph in document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS):
        style = _docx_paragraph_style(paragraph)
        if style not in {"Heading1", "Heading2", "Heading3"}:
            continue
        ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
        if ppr is None:
            ppr = etree.Element(_w("pPr"))
            paragraph.insert(0, ppr)
        existing = ppr.find("./w:numPr", namespaces=_DOCX_NS)
        if existing is not None:
            ppr.remove(existing)
        numpr = etree.SubElement(ppr, _w("numPr"))
        ilvl = etree.SubElement(numpr, _w("ilvl"))
        ilvl.set(_w("val"), str({"Heading1": 0, "Heading2": 1, "Heading3": 2}[style]))
        numid = etree.SubElement(numpr, _w("numId"))
        numid.set(_w("val"), num_id)
        applied = True
    return applied


def _remove_empty_docx_paragraphs(document_root: etree._Element) -> int:
    removed = 0
    for paragraph in list(document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS)):
        style = _docx_paragraph_style(paragraph)
        if style in {"Title", "Author", "Abstract", "ImageCaption", "TableCaption"}:
            continue
        if _docx_paragraph_has_payload(paragraph):
            continue
        parent = paragraph.getparent()
        if parent is not None:
            parent.remove(paragraph)
            removed += 1
    return removed


def _apply_bibliography_style(document_root: etree._Element) -> bool:
    paragraphs = document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS)
    in_bibliography = False
    changed = False
    for paragraph in paragraphs:
        text = _docx_paragraph_text(paragraph)
        style = _docx_paragraph_style(paragraph)
        if text.lower() in {"references", "bibliography"} and style in {"Heading1", "Heading2", "Heading3", None}:
            in_bibliography = True
            continue
        if in_bibliography and style not in {"Heading1", "Heading2", "Heading3"} and text:
            ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
            if ppr is None:
                ppr = etree.Element(_w("pPr"))
                paragraph.insert(0, ppr)
            style_node = ppr.find("./w:pStyle", namespaces=_DOCX_NS)
            if style_node is None:
                style_node = etree.SubElement(ppr, _w("pStyle"))
            style_node.set(_w("val"), "Bibliography")
            changed = True
    return changed


def _style_inline_numeric_citations(document_root: etree._Element) -> int:
    citation_re = re.compile(r"^\[\d+(?:,\s*\d+)*\]$")
    styled = 0
    for paragraph in document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS):
        style = _docx_paragraph_style(paragraph)
        if style not in {"BodyText", "FirstParagraph", "Compact"}:
            continue
        for run in paragraph.findall("./w:r", namespaces=_DOCX_NS):
            text = "".join(run.xpath(".//w:t/text()", namespaces=_DOCX_NS)).strip()
            if not citation_re.fullmatch(text):
                continue
            rpr = run.find("./w:rPr", namespaces=_DOCX_NS)
            if rpr is None:
                rpr = etree.Element(_w("rPr"))
                run.insert(0, rpr)
            vert = rpr.find("./w:vertAlign", namespaces=_DOCX_NS)
            if vert is None:
                vert = etree.SubElement(rpr, _w("vertAlign"))
            vert.set(_w("val"), "superscript")
            for tag in ("sz", "szCs"):
                size = rpr.find(f"./w:{tag}", namespaces=_DOCX_NS)
                if size is None:
                    size = etree.SubElement(rpr, _w(tag))
                size.set(_w("val"), "16")
            styled += 1
    return styled


def _normalize_references_heading(document_root: etree._Element) -> bool:
    changed = False
    for paragraph in document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS):
        if _docx_paragraph_text(paragraph).strip().lower() != "references":
            continue
        ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
        if ppr is None:
            continue
        numpr = ppr.find("./w:numPr", namespaces=_DOCX_NS)
        if numpr is not None:
            ppr.remove(numpr)
            changed = True
    return changed


def _scale_captioned_figures(document_root: etree._Element) -> int:
    target_width = 5800000
    max_height = 3400000
    scaled = 0
    for paragraph in document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS):
        if _docx_paragraph_style(paragraph) != "CaptionedFigure":
            continue
        for inline in paragraph.xpath(".//wp:inline", namespaces={**_DOCX_NS, "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}):
            extent = inline.find("./wp:extent", namespaces={**_DOCX_NS, "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"})
            xfrm_ext = inline.find(".//a:xfrm/a:ext", namespaces={**_DOCX_NS, "a": "http://schemas.openxmlformats.org/drawingml/2006/main"})
            if extent is None:
                continue
            try:
                cx = int(extent.get("cx", "0"))
                cy = int(extent.get("cy", "0"))
            except ValueError:
                continue
            if cx <= 0 or cy <= 0:
                continue
            new_cx = target_width
            new_cy = int(cy * (new_cx / cx))
            if new_cy > max_height:
                ratio = max_height / new_cy
                new_cy = max_height
                new_cx = int(new_cx * ratio)
            if new_cx <= cx and new_cy <= cy:
                continue
            extent.set("cx", str(new_cx))
            extent.set("cy", str(new_cy))
            if xfrm_ext is not None:
                xfrm_ext.set("cx", str(new_cx))
                xfrm_ext.set("cy", str(new_cy))
            scaled += 1
    return scaled


def _ensure_tbl_child(parent: etree._Element, name: str) -> etree._Element:
    child = parent.find(f"./w:{name}", namespaces=_DOCX_NS)
    if child is None:
        child = etree.SubElement(parent, _w(name))
    return child


def _set_border(node: etree._Element, edge: str, *, val: str, sz: int, space: int = 0) -> None:
    border = node.find(f"./w:{edge}", namespaces=_DOCX_NS)
    if border is None:
        border = etree.SubElement(node, _w(edge))
    border.set(_w("val"), val)
    border.set(_w("sz"), str(sz))
    border.set(_w("space"), str(space))
    border.set(_w("color"), "000000")


def _cell_text(cell: etree._Element) -> str:
    return "".join(cell.xpath(".//w:t/text()", namespaces=_DOCX_NS)).strip()


def _is_numeric_table_text(text: str) -> bool:
    compact = text.replace(",", "").replace("%", "").strip()
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+|N/A)", compact, re.IGNORECASE))


def _set_paragraph_alignment(paragraph: etree._Element, align: str) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.Element(_w("pPr"))
        paragraph.insert(0, ppr)
    jc = ppr.find("./w:jc", namespaces=_DOCX_NS)
    if jc is None:
        jc = etree.SubElement(ppr, _w("jc"))
    jc.set(_w("val"), align)


def _set_paragraph_style(paragraph: etree._Element, style_id: str) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.Element(_w("pPr"))
        paragraph.insert(0, ppr)
    pstyle = ppr.find("./w:pStyle", namespaces=_DOCX_NS)
    if pstyle is None:
        pstyle = etree.SubElement(ppr, _w("pStyle"))
    pstyle.set(_w("val"), style_id)


def _set_paragraph_indent(
    paragraph: etree._Element,
    *,
    left: int = 0,
    first_line: int | None = None,
    hanging: int | None = None,
) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.Element(_w("pPr"))
        paragraph.insert(0, ppr)
    ind = ppr.find("./w:ind", namespaces=_DOCX_NS)
    if ind is None:
        ind = etree.SubElement(ppr, _w("ind"))
    ind.set(_w("left"), str(left))
    if first_line is None:
        ind.attrib.pop(_w("firstLine"), None)
    else:
        ind.set(_w("firstLine"), str(first_line))
    if hanging is None:
        ind.attrib.pop(_w("hanging"), None)
    else:
        ind.set(_w("hanging"), str(hanging))


def _set_paragraph_tabs(
    paragraph: etree._Element,
    *,
    center: int | None = None,
    right: int | None = None,
) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.Element(_w("pPr"))
        paragraph.insert(0, ppr)
    tabs = ppr.find("./w:tabs", namespaces=_DOCX_NS)
    if center is None and right is None:
        if tabs is not None:
            ppr.remove(tabs)
        return
    if tabs is None:
        tabs = etree.SubElement(ppr, _w("tabs"))
    else:
        for child in list(tabs):
            tabs.remove(child)
    for value, position in (("center", center), ("right", right)):
        if position is None:
            continue
        tab = etree.SubElement(tabs, _w("tab"))
        tab.set(_w("val"), value)
        tab.set(_w("pos"), str(position))


def _clear_paragraph_tabs(paragraph: etree._Element) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        return
    tabs = ppr.find("./w:tabs", namespaces=_DOCX_NS)
    if tabs is not None:
        ppr.remove(tabs)


def _set_paragraph_text(paragraph: etree._Element, text: str) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    for child in list(paragraph):
        if ppr is not None and child is ppr:
            continue
        paragraph.remove(child)
    run = etree.SubElement(paragraph, _w("r"))
    text_node = etree.SubElement(run, _w("t"))
    text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text


def _set_runs_bold(paragraph: etree._Element, *, enabled: bool) -> None:
    for run in paragraph.findall("./w:r", namespaces=_DOCX_NS):
        rpr = run.find("./w:rPr", namespaces=_DOCX_NS)
        if rpr is None:
            rpr = etree.SubElement(run, _w("rPr"))
        bold = rpr.find("./w:b", namespaces=_DOCX_NS)
        if enabled and bold is None:
            etree.SubElement(rpr, _w("b"))
        if not enabled and bold is not None:
            rpr.remove(bold)


def _set_runs_size(paragraph: etree._Element, size: int) -> None:
    for run in paragraph.findall("./w:r", namespaces=_DOCX_NS):
        rpr = run.find("./w:rPr", namespaces=_DOCX_NS)
        if rpr is None:
            rpr = etree.SubElement(run, _w("rPr"))
        sz = rpr.find("./w:sz", namespaces=_DOCX_NS)
        if sz is None:
            sz = etree.SubElement(rpr, _w("sz"))
        sz.set(_w("val"), str(size))
        szcs = rpr.find("./w:szCs", namespaces=_DOCX_NS)
        if szcs is None:
            szcs = etree.SubElement(rpr, _w("szCs"))
        szcs.set(_w("val"), str(size))


def _set_paragraph_spacing(paragraph: etree._Element, *, before: int, after: int, line: int | None = None) -> None:
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        ppr = etree.Element(_w("pPr"))
        paragraph.insert(0, ppr)
    spacing = ppr.find("./w:spacing", namespaces=_DOCX_NS)
    if spacing is None:
        spacing = etree.SubElement(ppr, _w("spacing"))
    spacing.set(_w("before"), str(before))
    spacing.set(_w("after"), str(after))
    if line is not None:
        spacing.set(_w("line"), str(line))
        spacing.set(_w("lineRule"), "auto")


def _ensure_math_para_centered(paragraph: etree._Element) -> None:
    for math_para in paragraph.findall("./m:oMathPara", namespaces=_DOCX_NS):
        math_para_pr = math_para.find("./m:oMathParaPr", namespaces=_DOCX_NS)
        if math_para_pr is None:
            math_para_pr = etree.Element(f"{{{_M_NS}}}oMathParaPr")
            math_para.insert(0, math_para_pr)
        jc = math_para_pr.find("./m:jc", namespaces=_DOCX_NS)
        if jc is None:
            jc = etree.SubElement(math_para_pr, f"{{{_M_NS}}}jc")
        jc.set(f"{{{_M_NS}}}val", "center")


def _remove_equation_number_runs(paragraph: etree._Element) -> None:
    for run in list(paragraph.findall("./w:r", namespaces=_DOCX_NS)):
        run_text = "".join(run.xpath(".//w:t/text()", namespaces=_DOCX_NS)).strip()
        has_tab = run.find("./w:tab", namespaces=_DOCX_NS) is not None
        if re.fullmatch(r"\(\d+\)", run_text):
            paragraph.remove(run)
            continue
        if has_tab and (not run_text or re.fullmatch(r"\(\d+\)", run_text)):
            paragraph.remove(run)


def _normalize_display_equation_paragraph(
    paragraph: etree._Element,
    *,
    center_tab: int,
    right_tab: int,
) -> None:
    _remove_equation_number_runs(paragraph)
    _set_paragraph_alignment(paragraph, "left")
    _set_paragraph_indent(paragraph, left=0, first_line=0)
    _set_paragraph_spacing(paragraph, before=0, after=0)
    _set_paragraph_tabs(paragraph, center=center_tab, right=right_tab)


def _paragraph_contains_display_math(paragraph: etree._Element) -> bool:
    return bool(
        paragraph.xpath("./m:oMathPara | ./m:oMath", namespaces=_DOCX_NS)
    )


def _display_equation_paragraph_text(paragraph: etree._Element) -> str:
    kept: list[str] = []
    for run in paragraph.findall("./w:r", namespaces=_DOCX_NS):
        text = "".join(run.xpath(".//w:t/text()", namespaces=_DOCX_NS)).strip()
        has_tab = run.find("./w:tab", namespaces=_DOCX_NS) is not None
        if has_tab and not text:
            continue
        if re.fullmatch(r"\(\d+\)", text):
            continue
        if text:
            kept.append(text)
    return " ".join(kept).strip()


def _is_display_equation_paragraph(paragraph: etree._Element) -> bool:
    if not _paragraph_contains_display_math(paragraph):
        return False
    return _display_equation_paragraph_text(paragraph) == ""


def _inline_display_math(paragraph: etree._Element) -> None:
    for child in list(paragraph):
        if child.tag != f"{{{_M_NS}}}oMathPara":
            continue
        insert_at = paragraph.index(child)
        moved = False
        for math_node in list(child):
            if math_node.tag != f"{{{_M_NS}}}oMath":
                continue
            child.remove(math_node)
            paragraph.insert(insert_at, math_node)
            insert_at += 1
            moved = True
        if moved:
            paragraph.remove(child)


def _make_tab_run() -> etree._Element:
    run = etree.Element(_w("r"))
    etree.SubElement(run, _w("tab"))
    return run


def _make_text_run(text: str) -> etree._Element:
    run = etree.Element(_w("r"))
    text_node = etree.SubElement(run, _w("t"))
    text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return run


def _layout_display_equation_paragraph(
    paragraph: etree._Element,
    marker: str,
    *,
    center_tab: int,
    right_tab: int,
) -> bool:
    _normalize_display_equation_paragraph(
        paragraph,
        center_tab=center_tab,
        right_tab=right_tab,
    )
    _inline_display_math(paragraph)
    if not _paragraph_contains_display_math(paragraph):
        return False
    math_nodes = [
        child
        for child in list(paragraph)
        if child.tag == f"{{{_M_NS}}}oMath"
    ]
    if not math_nodes:
        return False
    first_index = paragraph.index(math_nodes[0])
    paragraph.insert(first_index, _make_tab_run())
    last_index = paragraph.index(math_nodes[-1]) + 1
    paragraph.insert(last_index, _make_tab_run())
    paragraph.insert(last_index + 1, _make_text_run(marker))
    return True


def _set_cell_width(cell: etree._Element, width: int) -> None:
    tcpr = _ensure_tbl_child(cell, "tcPr")
    tcw = _ensure_tbl_child(tcpr, "tcW")
    tcw.set(_w("type"), "dxa")
    tcw.set(_w("w"), str(width))


def _is_equation_layout_table(table: etree._Element) -> bool:
    caption = table.find("./w:tblPr/w:tblCaption", namespaces=_DOCX_NS)
    return bool(caption is not None and caption.get(_w("val")) == _DOCX_EQUATION_LAYOUT_CAPTION)


def _build_equation_layout_table(paragraph: etree._Element, marker: str) -> etree._Element:
    _normalize_display_equation_paragraph(
        paragraph,
        center_tab=4860,
        right_tab=9720,
    )

    table = etree.Element(_w("tbl"))
    tbl_pr = etree.SubElement(table, _w("tblPr"))
    tbl_caption = etree.SubElement(tbl_pr, _w("tblCaption"))
    tbl_caption.set(_w("val"), _DOCX_EQUATION_LAYOUT_CAPTION)

    tbl_w = etree.SubElement(tbl_pr, _w("tblW"))
    tbl_w.set(_w("type"), "dxa")
    tbl_w.set(_w("w"), "9360")

    tbl_layout = etree.SubElement(tbl_pr, _w("tblLayout"))
    tbl_layout.set(_w("type"), "fixed")

    tbl_jc = etree.SubElement(tbl_pr, _w("jc"))
    tbl_jc.set(_w("val"), "center")

    tbl_cell_mar = etree.SubElement(tbl_pr, _w("tblCellMar"))
    for edge in ("top", "bottom", "left", "right"):
        mar = etree.SubElement(tbl_cell_mar, _w(edge))
        mar.set(_w("w"), "0")
        mar.set(_w("type"), "dxa")

    tbl_borders = etree.SubElement(tbl_pr, _w("tblBorders"))
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        border = etree.SubElement(tbl_borders, _w(edge))
        border.set(_w("val"), "nil")
        border.set(_w("sz"), "0")
        border.set(_w("space"), "0")
        border.set(_w("color"), "000000")

    tbl_look = etree.SubElement(tbl_pr, _w("tblLook"))
    tbl_look.set(_w("firstRow"), "0")
    tbl_look.set(_w("lastRow"), "0")
    tbl_look.set(_w("firstColumn"), "0")
    tbl_look.set(_w("lastColumn"), "0")
    tbl_look.set(_w("noHBand"), "1")
    tbl_look.set(_w("noVBand"), "1")
    tbl_look.set(_w("val"), "0000")

    widths = [1440, 6480, 1440]
    tbl_grid = etree.SubElement(table, _w("tblGrid"))
    for width in widths:
        grid_col = etree.SubElement(tbl_grid, _w("gridCol"))
        grid_col.set(_w("w"), str(width))

    row = etree.SubElement(table, _w("tr"))
    trpr = etree.SubElement(row, _w("trPr"))
    etree.SubElement(trpr, _w("cantSplit"))

    left_cell = etree.SubElement(row, _w("tc"))
    _set_cell_width(left_cell, widths[0])
    _set_cell_v_align(left_cell, "center")
    left_paragraph = etree.SubElement(left_cell, _w("p"))
    _set_paragraph_spacing(left_paragraph, before=0, after=0)

    center_cell = etree.SubElement(row, _w("tc"))
    _set_cell_width(center_cell, widths[1])
    _set_cell_v_align(center_cell, "center")
    center_cell.append(paragraph)

    right_cell = etree.SubElement(row, _w("tc"))
    _set_cell_width(right_cell, widths[2])
    _set_cell_v_align(right_cell, "center")
    number_paragraph = etree.SubElement(right_cell, _w("p"))
    _set_paragraph_alignment(number_paragraph, "right")
    _set_paragraph_indent(number_paragraph, left=0, first_line=0)
    _set_paragraph_spacing(number_paragraph, before=0, after=0)
    _set_paragraph_text(number_paragraph, marker)

    return table


def _equation_layout_table_ok(table: etree._Element, marker: str | None = None) -> bool:
    _ = marker
    return _is_equation_layout_table(table)


def _display_equation_paragraph_aligned(paragraph: etree._Element) -> bool:
    if not _is_display_equation_paragraph(paragraph):
        return True
    ppr = paragraph.find("./w:pPr", namespaces=_DOCX_NS)
    if ppr is None:
        return False
    if paragraph.xpath("./m:oMathPara", namespaces=_DOCX_NS):
        return False
    jc = ppr.find("./w:jc", namespaces=_DOCX_NS)
    if jc is not None and jc.get(_w("val")) not in {None, "left"}:
        return False
    tabs = ppr.find("./w:tabs", namespaces=_DOCX_NS)
    if tabs is None:
        return False
    tab_values = [
        (tab.get(_w("val")), tab.get(_w("pos")))
        for tab in tabs.findall("./w:tab", namespaces=_DOCX_NS)
    ]
    if len(tab_values) < 2 or tab_values[0][0] != "center" or tab_values[1][0] != "right":
        return False
    ind = ppr.find("./w:ind", namespaces=_DOCX_NS)
    if ind is None:
        return False
    if ind.get(_w("left")) != "0" or ind.get(_w("firstLine")) != "0":
        return False
    tab_runs = paragraph.xpath("./w:r[w:tab]", namespaces=_DOCX_NS)
    if len(tab_runs) < 2:
        return False
    markers = [
        "".join(run.xpath(".//w:t/text()", namespaces=_DOCX_NS)).strip()
        for run in paragraph.findall("./w:r", namespaces=_DOCX_NS)
    ]
    return any(re.fullmatch(r"\(\d+\)", marker) for marker in markers)


def _display_equation_alignment_ok(document_root: etree._Element) -> bool:
    body = document_root.find("./w:body", namespaces=_DOCX_NS)
    if body is None:
        return True
    for child in body:
        if child.tag == _w("tbl") and _is_equation_layout_table(child):
            return False
        if child.tag == _w("p") and _is_display_equation_paragraph(child):
            if not _display_equation_paragraph_aligned(child):
                return False
    return True


def _docx_display_math_omml_ok(document_root: etree._Element) -> bool:
    for paragraph in document_root.xpath(".//w:body//w:p", namespaces=_DOCX_NS):
        text = _docx_paragraph_text(paragraph)
        if any(marker in text for marker in ("$$", "\\begin{aligned}", "\\end{aligned}", "\\[")):
            return False
    return True


def _docx_table_caption_numbering_ok(document_root: etree._Element) -> bool:
    body = document_root.find("./w:body", namespaces=_DOCX_NS)
    if body is None:
        return True
    table_number = 0
    children = list(body)
    for idx, child in enumerate(children):
        if child.tag != _w("tbl"):
            continue
        table_number += 1
        prev_idx = idx - 1
        while prev_idx >= 0 and children[prev_idx].tag == _w("p") and not _docx_paragraph_has_payload(children[prev_idx]):
            prev_idx -= 1
        if prev_idx < 0 or children[prev_idx].tag != _w("p"):
            return False
        caption_paragraph = children[prev_idx]
        if _docx_paragraph_style(caption_paragraph) != "TableCaption":
            return False
        caption_text = _normalize_docx_caption_text(_docx_paragraph_text(caption_paragraph))
        if not (
            re.match(rf"^Table\s+{table_number}[.:]?\s+", caption_text, re.IGNORECASE)
            or caption_text == f"Table {table_number}."
        ):
            return False
    return True


def _normalize_docx_figure_caption_text(text: str, figure_number: int) -> str:
    normalized = _normalize_docx_caption_text(text)
    if not normalized:
        return f"Figure {figure_number}."
    if re.match(r"^Figure\s+\d+[.:]?\s*", normalized, re.IGNORECASE):
        return normalized
    return f"Figure {figure_number}. {normalized}"


def _docx_figure_caption_numbering_ok(document_root: etree._Element) -> bool:
    paragraphs = list(document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS))
    figure_counter = 0
    for idx, paragraph in enumerate(paragraphs):
        if _docx_paragraph_style(paragraph) != "CaptionedFigure":
            continue
        figure_counter += 1
        matched_caption = False
        for next_paragraph in paragraphs[idx + 1 :]:
            next_style = _docx_paragraph_style(next_paragraph)
            if next_style == "ImageCaption":
                caption_text = _normalize_docx_caption_text(
                    _docx_paragraph_text(next_paragraph)
                )
                matched_caption = bool(
                    re.match(
                        rf"^Figure\s+{figure_counter}[.:]?\s+",
                        caption_text,
                        re.IGNORECASE,
                    )
                    or caption_text == f"Figure {figure_counter}."
                )
                break
            if next_style in {
                "CaptionedFigure",
                "Heading1",
                "Heading2",
                "Heading3",
                "TableCaption",
            }:
                break
        if not matched_caption:
            return False
    return True


def _normalize_docx_figure_captions(document_root: etree._Element) -> bool:
    paragraphs = list(document_root.xpath(".//w:body/w:p", namespaces=_DOCX_NS))
    figure_counter = 0
    for idx, paragraph in enumerate(paragraphs):
        if _docx_paragraph_style(paragraph) != "CaptionedFigure":
            continue
        figure_counter += 1
        for next_paragraph in paragraphs[idx + 1 :]:
            next_style = _docx_paragraph_style(next_paragraph)
            if next_style == "ImageCaption":
                original = _docx_paragraph_text(next_paragraph)
                normalized = _normalize_docx_figure_caption_text(
                    original,
                    figure_counter,
                )
                if normalized != original:
                    _set_paragraph_text(next_paragraph, normalized)
                break
            if next_style in {
                "CaptionedFigure",
                "Heading1",
                "Heading2",
                "Heading3",
                "TableCaption",
            }:
                break
    return _docx_figure_caption_numbering_ok(document_root)


def _set_row_border(row: etree._Element, edge: str, *, val: str, sz: int) -> None:
    trpr = _ensure_tbl_child(row, "trPr")
    borders = _ensure_tbl_child(trpr, "tblBorders")
    _set_border(borders, edge, val=val, sz=sz)


def _set_cell_v_align(cell: etree._Element, align: str) -> None:
    tcpr = _ensure_tbl_child(cell, "tcPr")
    valign = _ensure_tbl_child(tcpr, "vAlign")
    valign.set(_w("val"), align)


def _estimate_table_column_widths(table: etree._Element, total_twips: int = 9360) -> list[int]:
    rows = table.findall("./w:tr", namespaces=_DOCX_NS)
    if not rows:
        return []
    col_count = max((len(row.findall("./w:tc", namespaces=_DOCX_NS)) for row in rows), default=0)
    if col_count == 0:
        return []
    max_lens = [6] * col_count
    for row in rows:
        cells = row.findall("./w:tc", namespaces=_DOCX_NS)
        for idx, cell in enumerate(cells[:col_count]):
            text = _cell_text(cell)
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text).strip()
            max_lens[idx] = max(max_lens[idx], min(len(normalized), 36))
    min_width = 720
    usable = max(total_twips - (min_width * col_count), col_count * 120)
    weight_sum = sum(max_lens) or col_count
    widths = [min_width + int(usable * weight / weight_sum) for weight in max_lens]
    diff = total_twips - sum(widths)
    if widths:
        widths[-1] += diff
    return widths


def _apply_table_column_widths(table: etree._Element, widths: list[int]) -> None:
    if not widths:
        return
    tbl_grid = table.find("./w:tblGrid", namespaces=_DOCX_NS)
    if tbl_grid is None:
        tbl_grid = etree.Element(_w("tblGrid"))
        insert_at = 1 if table.find("./w:tblPr", namespaces=_DOCX_NS) is not None else 0
        table.insert(insert_at, tbl_grid)
    else:
        for child in list(tbl_grid):
            tbl_grid.remove(child)
    for width in widths:
        grid_col = etree.SubElement(tbl_grid, _w("gridCol"))
        grid_col.set(_w("w"), str(width))

    rows = table.findall("./w:tr", namespaces=_DOCX_NS)
    for row in rows:
        cells = row.findall("./w:tc", namespaces=_DOCX_NS)
        for idx, cell in enumerate(cells[: len(widths)]):
            tcpr = _ensure_tbl_child(cell, "tcPr")
            tcw = _ensure_tbl_child(tcpr, "tcW")
            tcw.set(_w("type"), "dxa")
            tcw.set(_w("w"), str(widths[idx]))


def _style_docx_tables(document_root: etree._Element) -> int:
    styled = 0
    for table in document_root.xpath(".//w:body/w:tbl", namespaces=_DOCX_NS):
        if _is_equation_layout_table(table):
            continue
        tbl_pr = _ensure_tbl_child(table, "tblPr")
        tbl_w = _ensure_tbl_child(tbl_pr, "tblW")
        tbl_w.set(_w("type"), "pct")
        tbl_w.set(_w("w"), "5000")
        tbl_layout = _ensure_tbl_child(tbl_pr, "tblLayout")
        tbl_layout.set(_w("type"), "fixed")
        tbl_jc = _ensure_tbl_child(tbl_pr, "jc")
        tbl_jc.set(_w("val"), "center")
        tbl_cell_mar = _ensure_tbl_child(tbl_pr, "tblCellMar")
        for edge in ("top", "bottom", "left", "right"):
            mar = _ensure_tbl_child(tbl_cell_mar, edge)
            mar.set(_w("w"), "30")
            mar.set(_w("type"), "dxa")
        borders = _ensure_tbl_child(tbl_pr, "tblBorders")
        _set_border(borders, "top", val="single", sz=10)
        _set_border(borders, "bottom", val="single", sz=10)
        _set_border(borders, "insideH", val="nil", sz=0)
        for edge in ("left", "right", "insideV"):
            _set_border(borders, edge, val="nil", sz=0)
        tbl_look = _ensure_tbl_child(tbl_pr, "tblLook")
        tbl_look.set(_w("firstRow"), "1")
        tbl_look.set(_w("lastRow"), "0")
        tbl_look.set(_w("firstColumn"), "0")
        tbl_look.set(_w("lastColumn"), "0")
        tbl_look.set(_w("noHBand"), "1")
        tbl_look.set(_w("noVBand"), "1")
        tbl_look.set(_w("val"), "0000")
        _apply_table_column_widths(table, _estimate_table_column_widths(table))

        rows = table.findall("./w:tr", namespaces=_DOCX_NS)
        for row_index, row in enumerate(rows):
            trpr = _ensure_tbl_child(row, "trPr")
            _ensure_tbl_child(trpr, "cantSplit")
            if row_index == 0:
                _ensure_tbl_child(trpr, "tblHeader").set(_w("val"), "on")
                _set_row_border(row, "bottom", val="single", sz=8)
            elif row_index == len(rows) - 1:
                _set_row_border(row, "bottom", val="single", sz=10)
            for cell in row.findall("./w:tc", namespaces=_DOCX_NS):
                _set_cell_v_align(cell, "center")
                text = _cell_text(cell)
                paragraphs = cell.findall(".//w:p", namespaces=_DOCX_NS)
                for para in paragraphs:
                    _set_paragraph_style(para, "Compact")
                    _set_paragraph_spacing(para, before=0, after=0, line=200)
                    if row_index == 0:
                        _set_paragraph_alignment(para, "center")
                        _set_runs_bold(para, enabled=True)
                        _set_runs_size(para, 16)
                    elif _is_numeric_table_text(text):
                        _set_paragraph_alignment(para, "right")
                        _set_runs_bold(para, enabled=False)
                        _set_runs_size(para, 16)
                    else:
                        _set_paragraph_alignment(para, "left")
                        _set_runs_bold(para, enabled=False)
                        _set_runs_size(para, 16)
        styled += 1
    return styled


def _docx_text_width(document_root: etree._Element) -> int:
    page_width = 12240
    left_margin = 1260
    right_margin = 1260
    body = document_root.find("./w:body", namespaces=_DOCX_NS)
    if body is not None:
        sect = body.find("./w:sectPr", namespaces=_DOCX_NS)
        if sect is not None:
            pg_sz = sect.find("./w:pgSz", namespaces=_DOCX_NS)
            pg_mar = sect.find("./w:pgMar", namespaces=_DOCX_NS)
            if pg_sz is not None:
                try:
                    page_width = int(pg_sz.get(_w("w"), page_width))
                except (TypeError, ValueError):
                    page_width = 12240
            if pg_mar is not None:
                try:
                    left_margin = int(pg_mar.get(_w("left"), left_margin))
                    right_margin = int(pg_mar.get(_w("right"), right_margin))
                except (TypeError, ValueError):
                    left_margin = 1260
                    right_margin = 1260
    return max(3600, page_width - left_margin - right_margin)


def _extract_equation_paragraph_from_layout_table(table: etree._Element) -> etree._Element | None:
    cells = table.findall("./w:tr/w:tc", namespaces=_DOCX_NS)
    if len(cells) != 3:
        return None
    paragraph = cells[1].find("./w:p", namespaces=_DOCX_NS)
    if paragraph is None:
        return None
    cells[1].remove(paragraph)
    return paragraph


def _append_display_equation_numbers(document_root: etree._Element) -> int:
    body = document_root.find("./w:body", namespaces=_DOCX_NS)
    if body is None:
        return 0
    text_width = _docx_text_width(document_root)
    center_tab = text_width // 2
    right_tab = text_width
    numbered = 0
    for child in list(body):
        if child.tag == _w("p") and _is_display_equation_paragraph(child):
            numbered += 1
            marker = f"({numbered})"
            _layout_display_equation_paragraph(
                child,
                marker,
                center_tab=center_tab,
                right_tab=right_tab,
            )
            continue
        if child.tag == _w("tbl") and _is_equation_layout_table(child):
            numbered += 1
            marker = f"({numbered})"
            paragraph = _extract_equation_paragraph_from_layout_table(child)
            if paragraph is None:
                continue
            if not _layout_display_equation_paragraph(
                paragraph,
                marker,
                center_tab=center_tab,
                right_tab=right_tab,
            ):
                continue
            insert_at = body.index(child)
            body.insert(insert_at, paragraph)
            body.remove(child)
    return numbered


def _postprocess_editorial_docx(docx_path: Path) -> dict[str, object]:
    quality: dict[str, object] = {
        "clean": False,
        "heading_numbering_ok": False,
        "removed_empty_paragraphs": 0,
        "styled_tables": 0,
        "equation_numbers_present": False,
        "equation_alignment_ok": False,
        "display_math_omml_ok": False,
        "keywords_present": False,
        "numeric_citations_plain": False,
        "figure_caption_numbering_ok": False,
        "table_caption_numbering_ok": False,
        "issues": [],
    }
    if not docx_path.exists():
        quality["issues"] = ["missing_docx"]
        return quality
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            files = {name: zf.read(name) for name in zf.namelist()}
        document_root = etree.fromstring(files["word/document.xml"])
        styles_root = etree.fromstring(files["word/styles.xml"])
        numbering_root = etree.fromstring(
            files.get(
                "word/numbering.xml",
                b'<?xml version="1.0" encoding="UTF-8"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
            )
        )
        num_id = _ensure_docx_heading_numbering(numbering_root)
        heading_numbering_ok = _apply_docx_heading_numbering(document_root, num_id)
        removed = _remove_empty_docx_paragraphs(document_root)
        _apply_bibliography_style(document_root)
        _style_inline_numeric_citations(document_root)
        _normalize_references_heading(document_root)
        styled_tables = _style_docx_tables(document_root)
        equation_numbers = _append_display_equation_numbers(document_root)
        equation_alignment_ok = _display_equation_alignment_ok(document_root)
        display_math_omml_ok = _docx_display_math_omml_ok(document_root)
        _scale_captioned_figures(document_root)
        figure_caption_numbering_ok = _normalize_docx_figure_captions(document_root)
        table_caption_numbering_ok = _docx_table_caption_numbering_ok(document_root)
        _apply_docx_page_layout(document_root)
        _normalize_docx_styles(styles_root)
        keywords_present = bool(
            document_root.xpath(
                './/w:p[w:pPr/w:pStyle[@w:val="Keywords"]]',
                namespaces=_DOCX_NS,
            )
        )
        numeric_citations_plain = not bool(
            document_root.xpath(
                './/w:vertAlign[@w:val="superscript"]',
                namespaces=_DOCX_NS,
            )
        )
        issues: list[str] = []
        if not equation_alignment_ok:
            issues.append("equation_alignment_not_centered")
        if not display_math_omml_ok:
            issues.append("display_math_not_omml")
        if not figure_caption_numbering_ok:
            issues.append("figure_caption_numbering_missing")
        if not table_caption_numbering_ok:
            issues.append("table_caption_numbering_missing")
        files["word/document.xml"] = etree.tostring(
            document_root,
            encoding="UTF-8",
            xml_declaration=True,
        )
        files["word/styles.xml"] = etree.tostring(
            styles_root,
            encoding="UTF-8",
            xml_declaration=True,
        )
        files["word/numbering.xml"] = etree.tostring(
            numbering_root,
            encoding="UTF-8",
            xml_declaration=True,
        )
        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        quality.update(
            {
                "clean": not issues,
                "heading_numbering_ok": heading_numbering_ok,
                "removed_empty_paragraphs": removed,
                "styled_tables": styled_tables,
                "equation_numbers_present": equation_numbers > 0,
                "equation_alignment_ok": equation_alignment_ok,
                "display_math_omml_ok": display_math_omml_ok,
                "keywords_present": keywords_present,
                "numeric_citations_plain": numeric_citations_plain,
                "figure_caption_numbering_ok": figure_caption_numbering_ok,
                "table_caption_numbering_ok": table_caption_numbering_ok,
                "issues": issues,
            }
        )
        return quality
    except Exception as exc:  # noqa: BLE001
        quality["issues"] = [f"postprocess_failed:{exc}"]
        return quality


def _count_pdf_pages(pdf_path: Path) -> int:
    if not pdf_path.exists():
        return 0
    pdfinfo_bin = which("pdfinfo")
    if pdfinfo_bin:
        try:
            result = subprocess.run(
                [pdfinfo_bin, str(pdf_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            match = re.search(r"(?mi)^Pages:\s*(\d+)\s*$", output)
            if result.returncode == 0 and match:
                return int(match.group(1))
        except Exception:  # noqa: BLE001
            pass
    try:
        raw = pdf_path.read_bytes()
    except OSError:
        return 0
    return max(raw.count(b"/Type /Page"), 0)


def _convert_docx_to_pdf_for_page_count(docx_path: Path) -> Path | None:
    soffice_bin = which("soffice")
    if not soffice_bin or not docx_path.exists():
        return None
    output_dir = docx_path.parent / ".docx_page_count_pdf"
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    try:
        result = subprocess.run(
            [
                soffice_bin,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(docx_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    pdf_path = output_dir / f"{docx_path.stem}.pdf"
    return pdf_path if pdf_path.exists() else None


def _audit_editorial_constraints(
    stage_dir: Path,
    *,
    markdown: str,
    config: RCConfig,
) -> dict[str, object]:
    quality: dict[str, object] = {
        "reference_count": 0,
        "reference_limit_ok": True,
        "docx_page_count": 0,
        "docx_page_limit_ok": True,
        "issues": [],
        "warnings": [],
    }
    issues: list[str] = []
    warnings: list[str] = []

    citation_keys = _collect_citation_keys(markdown)
    quality["reference_count"] = len(citation_keys)
    if config.export.max_references > 0 and len(citation_keys) > config.export.max_references:
        quality["reference_limit_ok"] = False
        issues.append(
            f"reference_limit_exceeded:{len(citation_keys)}>{config.export.max_references}"
        )

    if config.export.docx_page_limit > 0:
        docx_path = stage_dir / "paper_repaired.docx"
        pdf_path = _convert_docx_to_pdf_for_page_count(docx_path)
        page_count = _count_pdf_pages(pdf_path) if pdf_path else 0
        quality["docx_page_count"] = page_count
        if page_count <= 0:
            warnings.append("docx_page_count_unavailable")
        elif page_count > config.export.docx_page_limit:
            quality["docx_page_limit_ok"] = False
            issues.append(
                f"docx_page_limit_exceeded:{page_count}>{config.export.docx_page_limit}"
            )

    quality["issues"] = issues
    quality["warnings"] = warnings
    return quality


def _pandoc_docx_citeproc_args(pandoc_bin: str) -> list[str]:
    try:
        help_result = subprocess.run(
            [pandoc_bin, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        return []
    help_text = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
    if "--citeproc" in help_text:
        return ["--citeproc"]
    citeproc_bin = which("pandoc-citeproc")
    if citeproc_bin:
        return ["--filter", citeproc_bin]
    return []


def _is_explanation_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(stripped):
        return False
    if _is_caption_block(stripped):
        return False
    if _IMAGE_RE.search(stripped):
        return False
    if stripped.startswith("|") or stripped.startswith(">"):
        return False
    lowered = stripped.lower()
    figure_cues = (
        "figure",
        "fig.",
        "shown",
        "below",
        "visual",
        "summar",
        "illustrat",
        "diagram",
    )
    return any(cue in lowered for cue in figure_cues)


def _is_explanation_for_bundle(
    block: str,
    figure_number: int | None,
    image_path: str,
    alt_text: str,
    caption_block: str | None,
) -> bool:
    stripped = block.strip()
    if not _is_explanation_block(stripped):
        return False
    lowered = stripped.lower()
    if "after the figure" in lowered:
        return False
    if figure_number is not None and re.search(rf"\bFigure\s+{figure_number}\b", stripped, re.IGNORECASE):
        return True
    if re.search(r"\bFigure\s+\d+\b", stripped, re.IGNORECASE):
        return True
    image_name = Path(image_path).name
    if image_name.startswith("pipeline_overview_") and "protocol" in lowered:
        return True
    if image_name.startswith("architecture_diagram_") and (
        "architecture" in lowered or "model" in lowered
    ):
        return True
    caption_text = _clean_caption_text(caption_block or "")
    if _keyword_overlap(stripped, caption_text) >= 2:
        return True
    if _keyword_overlap(stripped, alt_text) >= 2:
        return True
    return any(cue in lowered for cue in ("shown", "below", "summar", "illustrat", "visual"))


def _extract_bundles(blocks: list[str]) -> list[_Bundle]:
    sections = _section_contexts(blocks)
    bundles: list[_Bundle] = []
    for idx, block in enumerate(blocks):
        image_match = _IMAGE_RE.search(block)
        if not image_match:
            continue
        alt_text = _extract_image_alt_text(block)
        caption_index = None
        explanation_index = None
        start = idx
        end = idx
        if idx > 0 and _is_caption_block(blocks[idx - 1]):
            caption_index = idx - 1
            start = min(start, caption_index)
        elif idx + 1 < len(blocks) and _is_caption_block(blocks[idx + 1]):
            caption_index = idx + 1
            end = max(end, caption_index)
        figure_number = None
        caption_block = (
            blocks[caption_index]
            if caption_index is not None and 0 <= caption_index < len(blocks)
            else ""
        )
        number_match = re.search(r"Figure\s+(\d+)", caption_block, re.IGNORECASE)
        if number_match:
            figure_number = int(number_match.group(1))
        explanation_indices: list[int] = []
        if start > 0 and _is_explanation_for_bundle(
            blocks[start - 1],
            figure_number,
            image_match.group(1),
            alt_text,
            caption_block,
        ):
            explanation_index = start - 1
            explanation_indices.append(explanation_index)
            start = explanation_index
        trailing_idx = end + 1
        if trailing_idx < len(blocks) and _is_explanation_for_bundle(
            blocks[trailing_idx],
            figure_number,
            image_match.group(1),
            alt_text,
            caption_block,
        ):
            explanation_indices.append(trailing_idx)
            end = trailing_idx
        bundles.append(
            _Bundle(
                image_path=image_match.group(1),
                alt_text=alt_text,
                image_index=idx,
                start=start,
                end=end,
                caption_index=caption_index,
                explanation_indices=tuple(sorted(set(explanation_indices))),
                section=sections[idx],
                figure_number=figure_number,
            )
        )
    return bundles


def _find_first_explicit_figure_reference_index(
    blocks: list[str],
    figure_number: int,
) -> int | None:
    pattern = re.compile(rf"\bFigure\s+{figure_number}\b", re.IGNORECASE)
    for idx, block in enumerate(blocks):
        if _IMAGE_RE.search(block) or _is_caption_block(block):
            continue
        if pattern.search(block):
            return idx
    return None


def _find_first_figure_reference_index(
    blocks: list[str],
    figure_number: int,
) -> int | None:
    return _find_first_explicit_figure_reference_index(blocks, figure_number)


def _find_first_explicit_table_reference_index(
    blocks: list[str],
    table_number: int,
) -> int | None:
    pattern = re.compile(rf"\bTable\s+{table_number}\b", re.IGNORECASE)
    for idx, block in enumerate(blocks):
        stripped = block.strip()
        if stripped.startswith("|") or _TABLE_CAPTION_RE.match(stripped):
            continue
        if _IMAGE_RE.search(stripped) or _is_caption_block(stripped):
            continue
        if pattern.search(stripped):
            return idx
    return None


def _is_plain_paragraph(block: str) -> bool:
    stripped = block.strip()
    return bool(
        stripped
        and not _HEADING_RE.match(stripped)
        and not _IMAGE_RE.search(stripped)
        and not _is_caption_block(stripped)
        and not stripped.startswith("|")
        and not stripped.startswith(">")
    )


def _maybe_move_bundle_closer_to_reference(blocks: list[str]) -> tuple[list[str], list[str]]:
    moved: list[str] = []
    for bundle in _extract_bundles(blocks):
        if bundle.figure_number is None:
            continue
        ref_idx = _find_first_figure_reference_index(blocks, bundle.figure_number)
        if ref_idx is None or bundle.start - ref_idx <= 2:
            continue
        target_insert = ref_idx + 1
        if target_insert < len(blocks) and _is_plain_paragraph(blocks[target_insert]):
            target_insert += 1
        bundle_blocks = blocks[bundle.start : bundle.end + 1]
        reduced = blocks[: bundle.start] + blocks[bundle.end + 1 :]
        if bundle.start < target_insert:
            target_insert -= (bundle.end - bundle.start + 1)
        blocks = reduced[:target_insert] + bundle_blocks + reduced[target_insert:]
        moved.append(Path(bundle.image_path).name)
    return blocks, moved


def _clean_caption_text(block: str) -> str:
    text = block.strip()
    bold_match = _BOLD_FIGURE_CAPTION_RE.match(text)
    if bold_match:
        return bold_match.group(1).strip()
    italic_match = _ITALIC_FIGURE_CAPTION_RE.match(text)
    if italic_match:
        return italic_match.group(1).strip()
    plain_match = _PLAIN_FIGURE_CAPTION_RE.match(text)
    if plain_match:
        return re.sub(r"^Figure\s+\d+[.:]\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\*\*Figure\s+\d+[.:]?\s*\*\*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\*Figure\s+\d+[.:]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Figure\s+\d+[.:]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*$", "", text).strip()
    return text.replace("**", "").replace("*", "").strip()


def _stage_editorial_compile_inputs(stage_dir: Path, run_dir: Path) -> None:
    _copy_tree_contents(run_dir / "stage-22" / "charts", stage_dir / "charts")
    for candidate in (
        run_dir / "stage-23" / "references_verified.bib",
        run_dir / "stage-22" / "references.bib",
    ):
        if candidate.exists():
            shutil.copy2(candidate, stage_dir / "references.bib")
            break


def _sentence_case(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def _build_explanation(
    image_path: str,
    alt_text: str,
    caption_block: str | None,
    section: str,
) -> str:
    image_name = Path(image_path).name
    if image_name.startswith("pipeline_overview_"):
        return "The end-to-end evaluation protocol is summarized visually below."
    if image_name.startswith("architecture_diagram_"):
        return "The model architecture is summarized visually below."

    caption_text = _clean_caption_text(caption_block or "")
    if not caption_text:
        caption_text = alt_text.strip()
    if caption_text:
        if caption_text.endswith("."):
            caption_text = caption_text[:-1]
        return f"The figure below summarizes {_sentence_case(caption_text)}."
    if section.lower() == "results":
        return "The figure below summarizes the main empirical comparison."
    return "The figure below summarizes the local discussion."


def _find_section_indices(blocks: list[str], heading: str) -> tuple[int, int] | None:
    start = None
    for idx, block in enumerate(blocks):
        match = _HEADING_RE.match(block)
        if not match or len(match.group(1)) > 2:
            continue
        title = match.group(2).strip().lower()
        if start is None and title == heading.lower():
            start = idx
            continue
        if start is not None:
            return start, idx
    if start is None:
        return None
    return start, len(blocks)


def _insert_after_first_paragraph(
    blocks: list[str],
    section_range: tuple[int, int],
    bundle_blocks: list[str],
) -> list[str]:
    start, end = section_range
    insert_at = start + 1
    while insert_at < end and _HEADING_RE.match(blocks[insert_at]):
        insert_at += 1
    if insert_at < end and _is_explanation_block(blocks[insert_at]):
        insert_at += 1
    return blocks[:insert_at] + bundle_blocks + blocks[insert_at:]


def _move_pipeline_bundle_to_setup(blocks: list[str]) -> tuple[list[str], bool]:
    bundles = _extract_bundles(blocks)
    target = next(
        (
            bundle
            for bundle in bundles
            if Path(bundle.image_path).name.startswith("pipeline_overview_")
        ),
        None,
    )
    if target is None or target.section.lower() != "introduction":
        return blocks, False
    setup_range = _find_section_indices(blocks, "Setup")
    if setup_range is None:
        return blocks, False
    bundle_blocks = blocks[target.start : target.end + 1]
    if bundle_blocks and _is_explanation_block(bundle_blocks[0]):
        bundle_blocks = bundle_blocks[1:]
    bundle_blocks = [
        "The end-to-end evaluation protocol is summarized visually below.",
        *bundle_blocks,
    ]
    reduced = blocks[: target.start] + blocks[target.end + 1 :]
    adjusted_start = target.start
    adjusted_setup = setup_range
    if target.start < setup_range[0]:
        width = target.end - target.start + 1
        adjusted_setup = (setup_range[0] - width, setup_range[1] - width)
    moved = _insert_after_first_paragraph(reduced, adjusted_setup, bundle_blocks)
    return moved, True


def _add_missing_explanations(blocks: list[str]) -> tuple[list[str], int]:
    bundles = _extract_bundles(blocks)
    inserted = 0
    for bundle in reversed(bundles):
        if bundle.explanation_indices:
            continue
        caption_block = (
            blocks[bundle.caption_index]
            if bundle.caption_index is not None and bundle.caption_index < len(blocks)
            else None
        )
        explanation = _build_explanation(
            bundle.image_path,
            bundle.alt_text,
            caption_block,
            bundle.section,
        )
        blocks = blocks[: bundle.start] + [explanation] + blocks[bundle.start :]
        inserted += 1
    return blocks, inserted


def _audit_markdown(blocks: list[str]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for bundle in _extract_bundles(blocks):
        image_name = Path(bundle.image_path).name
        if not bundle.explanation_indices:
            issues.append(
                {
                    "type": "missing_explanation",
                    "severity": "high",
                    "image": image_name,
                    "section": bundle.section,
                }
            )
        if image_name.startswith("pipeline_overview_") and bundle.section.lower() == "introduction":
            issues.append(
                {
                    "type": "wrong_section_placement",
                    "severity": "high",
                    "image": image_name,
                    "section": bundle.section,
                    "target_section": "Setup",
                }
            )
        if bundle.figure_number is not None:
            ref_idx = _find_first_explicit_figure_reference_index(blocks, bundle.figure_number)
            if ref_idx is None:
                issues.append(
                    {
                        "type": "missing_explicit_figure_reference",
                        "severity": "high",
                        "image": image_name,
                        "section": bundle.section,
                        "figure_number": bundle.figure_number,
                    }
                )
            elif bundle.start - ref_idx > 2:
                issues.append(
                    {
                        "type": "far_from_first_reference",
                        "severity": "high",
                        "image": image_name,
                        "section": bundle.section,
                        "figure_number": bundle.figure_number,
                        "distance_blocks": bundle.start - ref_idx,
                    }
                )
    for idx, block in enumerate(blocks):
        caption = block.strip()
        if not _TABLE_CAPTION_RE.match(caption):
            continue
        table_number = _extract_table_number(caption)
        if table_number is None:
            continue
        if _find_first_explicit_table_reference_index(blocks, table_number) is None:
            issues.append(
                {
                    "type": "missing_explicit_table_reference",
                    "severity": "high",
                    "table_number": table_number,
                    "block_index": idx,
                }
            )
    return issues


def _extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _copy_tree_contents(src: Path, dest: Path) -> None:
    if not src.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dest / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _load_editorial_source(run_dir: Path) -> tuple[str, str]:
    candidates = (
        (run_dir / "stage-24" / "paper_repaired.md", "stage-24/paper_repaired.md"),
        (run_dir / "stage-23" / "paper_final_verified.md", "stage-23/paper_final_verified.md"),
        (run_dir / "stage-22" / "paper_final.md", "stage-22/paper_final.md"),
    )
    for path, label in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8"), label
    return "# Final Paper\n\nNo content generated.\n", "generated:fallback"


def _compile_editorial_tex(
    stage_dir: Path,
    repaired_markdown: str,
    config: RCConfig,
    *,
    run_dir: Path | None = None,
) -> tuple[list[str], list[str], bool]:
    try:
        from researchclaw.pipeline.stage_impls._review_publish import (
            _export_latex_pdf_artifacts,
        )

        return _export_latex_pdf_artifacts(
            stage_dir=stage_dir,
            run_dir=run_dir or stage_dir.parent,
            markdown=repaired_markdown,
            config=config,
            output_tex_name="paper_repaired.tex",
            output_pdf_name="paper_repaired.pdf",
            source_markdown_for_charts=repaired_markdown,
            artifacts_label="Stage 24",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 24: Editorial LaTeX generation skipped: %s", exc)
    return [], [], False


def _export_editorial_docx(
    stage_dir: Path,
    *,
    authors: str = "Anonymous",
    bibliography_name: str = "references.bib",
) -> tuple[list[str], list[str], bool]:
    artifacts: list[str] = []
    evidence: list[str] = []
    md_path = stage_dir / "paper_repaired.md"
    if not md_path.exists() or md_path.stat().st_size == 0:
        return artifacts, evidence, False
    pandoc_bin = which("pandoc")
    if not pandoc_bin:
        logger.warning("Stage 24: pandoc not available; skipping docx export")
        return artifacts, evidence, False
    reference_doc = _docx_reference_doc_path()
    if not reference_doc.exists():
        logger.warning("Stage 24: reference.docx missing; skipping docx export")
        return artifacts, evidence, False
    bibliography_text = ""
    bibliography_path = stage_dir / bibliography_name
    if bibliography_path.exists():
        bibliography_text = bibliography_path.read_text(encoding="utf-8")
    docx_md_path = stage_dir / "paper_repaired_docx.md"
    docx_md_path.write_text(
        _prepare_docx_markdown(
            md_path.read_text(encoding="utf-8"),
            authors=authors,
            bibliography_name=bibliography_name,
            bibliography_text=bibliography_text,
            rewrite_citations=False,
        ),
        encoding="utf-8",
    )
    docx_path = stage_dir / "paper_repaired.docx"
    cmd = [
        pandoc_bin,
        str(docx_md_path.name),
        "--standalone",
        "--from",
        "markdown+tex_math_dollars+tex_math_single_backslash",
        "--to",
        "docx",
        "--reference-doc",
        str(reference_doc),
        "--output",
        str(docx_path.name),
        "--resource-path",
        ".",
        "--wrap=none",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=stage_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 24: pandoc docx export failed: %s", exc)
        return artifacts, evidence, False
    if result.returncode != 0:
        logger.warning(
            "Stage 24: pandoc docx export failed (exit %d): %s",
            result.returncode,
            (result.stderr or "").strip()[:500],
        )
        return artifacts, evidence, False
    if docx_path.exists() and docx_path.stat().st_size > 0:
        quality = _postprocess_editorial_docx(docx_path)
        (stage_dir / "docx_quality.json").write_text(
            json.dumps(quality, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts.append("paper_repaired.docx")
        artifacts.append("docx_quality.json")
        evidence.append("stage-24/paper_repaired.docx")
        evidence.append("stage-24/docx_quality.json")
        return artifacts, evidence, True
    return artifacts, evidence, False


def _resolve_editorial_codex_binary(config: RCConfig) -> str | None:
    repair_cfg = config.experiment.editorial_repair
    if repair_cfg.provider != "codex_cli":
        return None
    if repair_cfg.binary_path:
        path = Path(repair_cfg.binary_path)
        return str(path) if path.exists() else None
    from shutil import which

    return which("codex")


def _copy_optional(src: Path, dest: Path) -> None:
    if src.exists() and src.is_file():
        shutil.copy2(src, dest)


def _build_editorial_task(
    *,
    issue_report: list[dict[str, object]],
    iteration: int,
    source_label: str,
    mode: str,
    submission_profile: str = "default",
) -> str:
    issues_json = json.dumps(issue_report, indent=2, ensure_ascii=False)
    mode_instruction = {
        "publish_first": (
            "Primary goal: make the paper read like a polished, submission-ready paper. "
            "You may add local explanation, transitions, and figure discussion even when the "
            "original draft is thin, as long as you do not change experimental facts."
        ),
        "balanced": (
            "Primary goal: improve readability and figure-text integration while staying close "
            "to the source draft."
        ),
        "conservative": (
            "Primary goal: fix local structural and wording issues without introducing new "
            "substantive narrative beyond what is already strongly supported by the draft."
        ),
    }.get(mode, "Primary goal: improve the final paper conservatively.")
    shared_content_rules = (
        "Shared content rules:\n"
        "- Ensure the abstract is followed by a **Keywords:** block.\n"
        "- Keep inline math inline; do not force inline symbols into display equations.\n"
        "- Every display equation must be written as a standalone equation block that can be numbered downstream.\n"
        "- Every display equation needs a natural lead-in sentence before it and concise prose for important symbols after it when needed.\n"
        "- Avoid mechanical `in Equation (n)` and `In Equation (n), ...` templates in the canonical markdown prose.\n"
        "- Every retained figure and table must be cited in nearby body text with an explicit `Figure N` or `Table N` reference.\n"
        "- Remove dangling commas or periods around display equations.\n\n"
    )
    profile_rules = ""
    if submission_profile == "ei_conference":
        profile_rules = (
            "EI conference profile rules:\n"
            "- Do not keep a standalone Related Work section; integrate necessary prior-work comparison into nearby sections.\n"
            "- Rename the main results chapter to `Results and Analysis`.\n"
            "- Fold Discussion and Limitations content into Conclusion.\n\n"
        )
    return (
        f"# Stage 24 Editorial Repair Task\n\n"
        f"Iteration: {iteration}\n"
        f"Source: {source_label}\n\n"
        "You are repairing a research paper markdown draft. Edit `paper_repaired.md` in place.\n\n"
        f"Mode:\n{mode_instruction}\n\n"
        "Goals:\n"
        "1. Fix figure placement and figure-text proximity issues.\n"
        "2. Ensure every retained figure and table has an explicit numbered body-text reference.\n"
        "3. Improve local flow around figures and clean obvious editorial rough edges.\n"
        "4. Fix ugly layout outcomes when they make the paper look unfinished, including single-figure pages, awkward page breaks, large blank areas around floats, and figures or tables that visibly break the opening of the next section.\n"
        "5. Write a structured `codex_review.json` describing remaining issues, risks, and whether another round is needed.\n\n"
        f"{shared_content_rules}"
        f"{profile_rules}"
        "Hard constraints:\n"
        "- Do not change experiment numbers, metric values, or table values.\n"
        "- Do not add or remove citation keys.\n"
        "- Do not invent new experiments or change conclusions' factual meaning.\n"
        "- Do not rewrite unrelated sections.\n"
        "- When fixing PDF layout issues, keep the target as normal LaTeX paper layout; do not imitate Word-style page layout in markdown.\n"
        "- Only edit `paper_repaired.md` and write `codex_review.json`.\n\n"
        "Allowed actions:\n"
        "- Move complete figure bundles closer to their first discussion.\n"
        "- Rewrite local paragraphs near problematic figures or awkward transitions.\n"
        "- Add specific figure explanations and bridging sentences.\n"
        "- Remove figures that are not consumed by the text.\n"
        "- Clean prompt-like captions.\n\n"
        "- Shorten overly long local figure discussion if it causes awkward page breaks.\n"
        "- Rewrite or tighten captions modestly when the current caption contributes to an obviously ugly layout.\n"
        "- shrink a figure modestly only when needed to avoid a single-figure page or visibly bad whitespace, and only if the figure remains readable.\n\n"
        "Review instructions:\n"
        "- Inspect the whole paper yourself, not just the detected issue list.\n"
        "- Prioritize issues that make the paper look unfinished or obviously broken.\n"
        "- When prior repaired TeX/PDF files are present in the workspace, use them to judge actual layout, not just markdown order.\n"
        "- In `codex_review.json`, include: issues, remaining_risks, should_continue, summary.\n\n"
        "Detected issues from local checks:\n"
        f"{issues_json}\n\n"
        "Required `codex_review.json` schema:\n"
        "{\n"
        '  "summary": "short string",\n'
        '  "issues": [{"type": "string", "severity": "critical|high|medium|low", "note": "string"}],\n'
        '  "remaining_risks": ["string"],\n'
        '  "should_continue": true\n'
        "}\n"
    )


def _prepare_codex_workspace(
    *,
    stage_dir: Path,
    run_dir: Path,
    source_markdown: str,
    issue_report: list[dict[str, object]],
    iteration: int,
    source_label: str,
    mode: str,
    submission_profile: str = "default",
    previous_review: dict[str, object] | None = None,
    previous_assessment: dict[str, object] | None = None,
) -> Path:
    stage_dir = stage_dir.resolve()
    workspace_root = (stage_dir / "codex_repair_workspace").resolve()
    workspace = (workspace_root / f"iter-{iteration}").resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "paper_input.md").write_text(source_markdown, encoding="utf-8")
    (workspace / "paper_repaired.md").write_text(source_markdown, encoding="utf-8")
    (workspace / "EDITORIAL_TASK.md").write_text(
        _build_editorial_task(
            issue_report=issue_report,
            iteration=iteration,
            source_label=source_label,
            mode=mode,
            submission_profile=submission_profile,
        ),
        encoding="utf-8",
    )
    (workspace / "editorial_issues.json").write_text(
        json.dumps(issue_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _copy_tree_contents(run_dir / "stage-22" / "charts", workspace / "charts")
    _copy_optional(run_dir / "stage-22" / "paper.tex", workspace / "paper.tex")
    _copy_optional(run_dir / "stage-24" / "paper_repaired.tex", workspace / "paper_repaired.tex")
    _copy_optional(run_dir / "stage-24" / "paper_repaired.pdf", workspace / "paper_repaired.pdf")
    _copy_optional(run_dir / "stage-24" / "paper_repaired.log", workspace / "paper_repaired.log")
    _copy_optional(
        run_dir / "stage-24" / "editorial_final_assessment.json",
        workspace / "editorial_final_assessment.json",
    )
    _copy_optional(
        run_dir / "stage-24" / "codex_review.json",
        workspace / "codex_review.json",
    )
    _copy_optional(
        run_dir / "stage-22" / "compilation_quality.json",
        workspace / "compilation_quality.json",
    )
    _copy_optional(
        run_dir / "stage-22" / "paper_verification.json",
        workspace / "paper_verification.json",
    )
    _copy_optional(run_dir / "stage-22" / "pdf_review.json", workspace / "pdf_review.json")
    _copy_optional(
        run_dir / "stage-23" / "verification_report.json",
        workspace / "verification_report.json",
    )
    if previous_review is not None:
        (workspace / "PREVIOUS_REVIEW.json").write_text(
            json.dumps(previous_review, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if previous_assessment is not None:
        (workspace / "PREVIOUS_ASSESSMENT.json").write_text(
            json.dumps(previous_assessment, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    for candidate in (
        run_dir / "stage-23" / "references_verified.bib",
        run_dir / "stage-22" / "references.bib",
    ):
        if candidate.exists():
            shutil.copy2(candidate, workspace / "references.bib")
            break
    return workspace


def _invoke_codex_editorial_round(
    *,
    workspace: Path,
    config: RCConfig,
    iteration: int,
) -> tuple[bool, str, dict[str, object] | None]:
    binary = _resolve_editorial_codex_binary(config)
    if not binary:
        return False, "Local codex CLI is not available", None

    repair_cfg = config.experiment.editorial_repair
    prompt = (
        "Read EDITORIAL_TASK.md and the local paper files. "
        "Repair paper_repaired.md in place and write codex_review.json. "
        "When finished, print a short summary of the edits."
    )
    codex_cmd = (
        f'cd "{workspace}" && '
        f'"{binary}" exec '
        f'{json.dumps(prompt)} '
        "--dangerously-bypass-approvals-and-sandbox "
        "--json "
        "-C ."
    )
    cmd = ["bash", "-lc", codex_cmd]
    if repair_cfg.model:
        cmd[2] += f" -m {json.dumps(repair_cfg.model)}"
    if repair_cfg.extra_args:
        cmd[2] += " " + " ".join(json.dumps(arg) for arg in repair_cfg.extra_args)

    (workspace / "codex_command.txt").write_text(cmd[2], encoding="utf-8")
    env_snapshot = {
        key: os.environ.get(key, "")
        for key in (
            "HOME",
            "PATH",
            "OPENAI_API_KEY",
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "all_proxy",
        )
    }
    (workspace / "codex_env.json").write_text(
        json.dumps(env_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=repair_cfg.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, f"Codex editorial repair timed out on iteration {iteration}", None
    elapsed = time.monotonic() - start

    (workspace / "codex_stdout.jsonl").write_text(result.stdout or "", encoding="utf-8")
    (workspace / "codex_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        return (
            False,
            f"Codex editorial repair failed on iteration {iteration} "
            f"(exit {result.returncode}, {elapsed:.1f}s): {(result.stderr or '').strip()[:500]}",
            None,
        )
    repaired_path = workspace / "paper_repaired.md"
    if not repaired_path.exists() or repaired_path.stat().st_size == 0:
        return False, "Codex did not produce paper_repaired.md", None
    review_path = workspace / "codex_review.json"
    if not review_path.exists() or review_path.stat().st_size == 0:
        return False, "Codex did not produce codex_review.json", None
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"Codex produced invalid codex_review.json: {exc}", None
    return True, repaired_path.read_text(encoding="utf-8"), review


def _extract_citation_keys(markdown: str) -> set[str]:
    keys: set[str] = set()
    for block in _split_blocks(markdown):
        if _is_caption_block(block) or _IMAGE_RE.search(block):
            continue
        for match in _CITATION_BRACKET_RE.finditer(block):
            for part in re.split(r"[,;]", match.group(1)):
                key = part.strip()
                if _looks_like_citation_key(key):
                    keys.add(key)
    return keys


def _looks_like_citation_key(text: str) -> bool:
    text = text.strip().lstrip("@")
    if not text or not re.fullmatch(r"[A-Za-z][A-Za-z0-9:_\-]*", text):
        return False
    # Most bib keys in this project are author-year style. Requiring a year-like
    # token prevents LaTeX float options such as [t] from being misclassified as
    # citation-key edits during Stage 24 boundary checks.
    return re.search(r"\d{4}", text) is not None


def _extract_table_numeric_tokens(markdown: str) -> list[str]:
    tokens: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            tokens.extend(_NUMBER_RE.findall(line))
    return tokens


def _extract_prose_numeric_tokens(markdown: str) -> list[str]:
    tokens: list[str] = []
    for block in _split_blocks(markdown):
        if _is_caption_block(block) or _IMAGE_RE.search(block):
            continue
        if block.strip().startswith("|"):
            continue
        tokens.extend(_NUMBER_RE.findall(block))
    return tokens


def _normalize_for_boundary_check(markdown: str, *, mode: str) -> str:
    normalized = markdown
    if mode == "publish_first":
        normalized = re.sub(
            r"\n\s*>\s*\*\*Note:\*\* This paper was produced in degraded mode\..*?(?=\n\s*\n|\Z)",
            "\n",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return normalized


def _audit_compiled_layout(stage_dir: Path) -> list[dict[str, object]]:
    tex_path = stage_dir / "paper_repaired.tex"
    if not tex_path.exists():
        return []
    tex = tex_path.read_text(encoding="utf-8")
    issues: list[dict[str, object]] = []
    section_matches = list(re.finditer(r"\\section\{(?P<section>[^}]+)\}", tex))
    float_matches = [
        *re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, flags=re.DOTALL),
        *re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, flags=re.DOTALL),
    ]
    float_matches.sort(key=lambda match: match.start())
    candidate_issues: list[dict[str, object]] = []
    for section_match in section_matches:
        section_start = section_match.start()
        prior_floats = [m for m in float_matches if m.end() <= section_start]
        if not prior_floats:
            continue
        float_match = prior_floats[-1]
        between = tex[float_match.end() : section_start]
        # Only treat the immediately preceding float as problematic when the
        # gap to the next section is very small and contains no other structure.
        if re.search(r"\\(?:sub)?section\{|\\begin\{(?:figure|table)\}", between):
            continue
        if len(re.sub(r"\s+", "", between)) > 400:
            continue
        float_block = float_match.group(0)
        caption_match = re.search(r"\\caption\{(.*?)\}", float_block, flags=re.DOTALL)
        label_match = re.search(r"\\label\{([^}]+)\}", float_block)
        caption = re.sub(r"\s+", " ", (caption_match.group(1) if caption_match else "")).strip()
        label = (label_match.group(1) if label_match else "").strip()
        target_hint = label or (caption[:80] if caption else "")
        candidate_issues.append(
            {
                "type": "awkward_float_layout",
                "severity": "high",
                "target_kind": "figure" if float_block.startswith("\\begin{figure}") else "table",
                "target_hint": target_hint,
                "section": section_match.group("section").strip(),
                "note": (
                    "A compiled float lands immediately before the next section heading, which is a strong "
                    "signal for awkward page breaks, near-standalone float pages, or a visually broken "
                    "section transition."
                ),
            }
        )
    if candidate_issues:
        issues.append(candidate_issues[-1])
    return issues


def _detect_boundary_violations(
    original_markdown: str,
    repaired_markdown: str,
    *,
    mode: str = "balanced",
) -> list[str]:
    violations: list[str] = []
    original_normalized = _normalize_for_boundary_check(original_markdown, mode=mode)
    repaired_normalized = _normalize_for_boundary_check(repaired_markdown, mode=mode)
    if _extract_citation_keys(original_normalized) != _extract_citation_keys(
        repaired_normalized
    ):
        violations.append("citation_keys_changed")
    return violations


def _run_codex_editorial_loop(
    stage_dir: Path,
    run_dir: Path,
    source_markdown: str,
    config: RCConfig,
) -> _CodexLoopResult:
    _stage_editorial_compile_inputs(stage_dir, run_dir)
    repair_cfg = config.experiment.editorial_repair
    source_label = _load_editorial_source(run_dir)[1]
    current_markdown = source_markdown
    issue_report = _audit_markdown(_split_blocks(current_markdown))
    review = {
        "source": source_label,
        "initial_issue_count": len(issue_report),
        "issues": issue_report,
    }
    iteration_log: list[dict[str, object]] = []
    previous_issue_count = len(issue_report)
    compile_clean = False
    latest_codex_review: dict[str, object] | None = None
    latest_assessment: dict[str, object] | None = None

    for iteration in range(1, max(1, repair_cfg.max_iterations) + 1):
        high_issues = [issue for issue in issue_report if issue.get("severity") == "high"]
        if iteration > 1 and not high_issues and repair_cfg.mode != "publish_first":
            break
        workspace = _prepare_codex_workspace(
            stage_dir=stage_dir,
            run_dir=run_dir,
            source_markdown=current_markdown,
            issue_report=issue_report,
            iteration=iteration,
            source_label=source_label,
            mode=repair_cfg.mode,
            submission_profile=config.export.submission_profile,
            previous_review=latest_codex_review,
            previous_assessment=latest_assessment,
        )
        ok, payload, codex_review = _invoke_codex_editorial_round(
            workspace=workspace,
            config=config,
            iteration=iteration,
        )
        if not ok:
            assessment = {
                "status": "fail",
                "remaining_issue_count": len(issue_report),
                "remaining_high_severity_issues": len(high_issues),
                "moved_figures": [],
                "rewritten_windows": [],
                "dropped_figures": [],
                "compile_clean": False,
                "improved_vs_stage22": False,
                "boundary_violations": [],
                "codex_reported_remaining_risks": [],
                "used_iterations": iteration,
            }
            iteration_log.append(
                {
                    "iteration": iteration,
                    "action": "codex_editorial_review_and_rewrite",
                    "changed": False,
                    "error": payload,
                }
            )
            return _CodexLoopResult(
                success=False,
                markdown=current_markdown,
                review=review,
                iterations=iteration_log,
                assessment=assessment,
                error=payload,
            )
        if codex_review is None:
            return _CodexLoopResult(
                success=False,
                markdown=current_markdown,
                review=review,
                iterations=iteration_log,
                assessment={
                    "status": "fail",
                    "remaining_issue_count": len(issue_report),
                    "remaining_high_severity_issues": len(high_issues),
                    "compile_clean": False,
                    "improved_vs_stage22": False,
                    "boundary_violations": [],
                    "codex_reported_remaining_risks": [],
                    "used_iterations": iteration,
                },
                error="Codex review output missing",
            )
        changed = payload.strip() != current_markdown.strip()
        boundary_violations = _detect_boundary_violations(
            current_markdown,
            payload,
            mode=repair_cfg.mode,
        )
        integrity_issues = _markdown_integrity_issues(payload, baseline=current_markdown)
        if boundary_violations:
            assessment = {
                "status": "fail",
                "remaining_issue_count": len(issue_report),
                "remaining_high_severity_issues": len(high_issues),
                "moved_figures": [],
                "rewritten_windows": [],
                "dropped_figures": [],
                "compile_clean": False,
                "improved_vs_stage22": False,
                "boundary_violations": boundary_violations,
                "codex_reported_remaining_risks": codex_review.get("remaining_risks", []),
                "used_iterations": iteration,
            }
            iteration_log.append(
                {
                    "iteration": iteration,
                    "action": "codex_editorial_review_and_rewrite",
                    "changed": changed,
                    "boundary_violations": boundary_violations,
                }
            )
            review["codex_review"] = codex_review
            return _CodexLoopResult(
                success=False,
                markdown=current_markdown,
                review=review,
                iterations=iteration_log,
                assessment=assessment,
                error=f"Stage 24 boundary violation: {', '.join(boundary_violations)}",
            )
        if integrity_issues:
            iteration_log.append(
                {
                    "iteration": iteration,
                    "action": "codex_editorial_review_and_rewrite",
                    "changed": changed,
                    "integrity_issues": integrity_issues,
                }
            )
            latest_codex_review = codex_review
            latest_assessment = {
                "status": "fail",
                "remaining_issue_count": len(issue_report),
                "remaining_high_severity_issues": len(high_issues),
                "compile_clean": False,
                "improved_vs_stage22": False,
                "boundary_violations": [],
                "integrity_issues": integrity_issues,
                "codex_reported_remaining_risks": codex_review.get("remaining_risks", []),
                "used_iterations": iteration,
            }
            review["current_issues"] = issue_report
            continue
        current_markdown = payload
        issue_report = _audit_markdown(_split_blocks(current_markdown))
        compile_artifacts, _, compile_clean = _compile_editorial_tex(
            stage_dir,
            current_markdown,
            config,
        )
        _ = compile_artifacts
        compiled_layout_issues = _audit_compiled_layout(stage_dir)
        if compiled_layout_issues:
            issue_report.extend(compiled_layout_issues)
        latest_codex_review = codex_review
        iteration_log.append(
            {
                "iteration": iteration,
                "action": "codex_editorial_review_and_rewrite",
                "changed": changed,
                "remaining_issue_count": len(issue_report),
                "codex_should_continue": bool(codex_review.get("should_continue", False)),
            }
        )
        current_high_issues = [issue for issue in issue_report if issue.get("severity") == "high"]
        review["current_issues"] = issue_report
        latest_assessment = {
            "status": "pass" if not current_high_issues else "warn",
            "remaining_issue_count": len(issue_report),
            "remaining_high_severity_issues": len(current_high_issues),
            "compile_clean": compile_clean,
            "improved_vs_stage22": len(issue_report) < review["initial_issue_count"],
            "boundary_violations": [],
            "codex_reported_remaining_risks": codex_review.get("remaining_risks", []),
            "remaining_issue_types": [str(issue.get("type", "")) for issue in issue_report],
            "remaining_issue_notes": [str(issue.get("note", "")) for issue in issue_report],
            "used_iterations": iteration,
        }
        if repair_cfg.mode == "publish_first":
            if (
                not bool(codex_review.get("should_continue", False))
                and compile_clean
                and not current_high_issues
            ):
                break
        if len(issue_report) >= previous_issue_count and not changed:
            break
        previous_issue_count = len(issue_report)

    high_remaining = [issue for issue in issue_report if issue.get("severity") == "high"]
    error = ""
    if high_remaining:
        error = (
            f"assessment failed: {len(high_remaining)} high-severity editorial issues remain"
        )
    codex_remaining_risks = []
    if latest_codex_review is not None:
        codex_remaining_risks = list(cast(list[object], latest_codex_review.get("remaining_risks", []))) if isinstance(latest_codex_review.get("remaining_risks", []), list) else []
    status = "pass"
    if high_remaining:
        status = "fail"
    elif compile_clean and codex_remaining_risks:
        status = "warn"
    assessment = {
        "status": status,
        "remaining_issue_count": len(issue_report),
        "remaining_high_severity_issues": len(high_remaining),
        "moved_figures": [],
        "rewritten_windows": [],
        "dropped_figures": [],
        "compile_clean": compile_clean,
        "improved_vs_stage22": len(issue_report) < review["initial_issue_count"],
        "boundary_violations": [],
        "codex_reported_remaining_risks": codex_remaining_risks,
        "remaining_issue_types": [str(issue.get("type", "")) for issue in issue_report],
        "remaining_issue_notes": [str(issue.get("note", "")) for issue in issue_report],
        "used_iterations": len(iteration_log),
    }
    if latest_codex_review is not None:
        review["codex_review"] = latest_codex_review
    review["final_issues"] = issue_report
    return _CodexLoopResult(
        success=not high_remaining,
        markdown=current_markdown,
        review=review,
        iterations=iteration_log,
        assessment=assessment,
        error=error,
    )


def _write_stage24_failure(
    stage_dir: Path,
    *,
    review: dict[str, object],
    iterations: list[dict[str, object]],
    assessment: dict[str, object],
) -> None:
    codex_review = review.get("codex_review")
    if not isinstance(codex_review, dict):
        codex_review = {
            "summary": "",
            "issues": [],
            "remaining_risks": [],
            "should_continue": False,
        }
    (stage_dir / "codex_review.json").write_text(
        json.dumps(codex_review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (stage_dir / "editorial_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (stage_dir / "editorial_iterations.json").write_text(
        json.dumps(iterations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (stage_dir / "editorial_final_assessment.json").write_text(
        json.dumps(assessment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _execute_final_editorial_repair(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> "StageResult":
    _ = adapters, llm, prompts
    from researchclaw.pipeline.executor import StageResult

    stage_dir.mkdir(parents=True, exist_ok=True)
    source_markdown, source_label = _load_editorial_source(run_dir)
    (stage_dir / "paper_editorial_input.md").write_text(source_markdown, encoding="utf-8")

    binary = _resolve_editorial_codex_binary(config)
    if not binary:
        initial_issues = _audit_markdown(_split_blocks(source_markdown))
        review = {
            "source": source_label,
            "initial_issue_count": len(initial_issues),
            "issues": initial_issues,
        }
        assessment = {
            "status": "fail",
            "remaining_issue_count": len(initial_issues),
            "remaining_high_severity_issues": len(
                [issue for issue in initial_issues if issue.get("severity") == "high"]
            ),
            "moved_figures": [],
            "rewritten_windows": [],
            "dropped_figures": [],
            "compile_clean": False,
            "improved_vs_stage22": False,
        }
        _write_stage24_failure(stage_dir, review=review, iterations=[], assessment=assessment)
        return StageResult(
            stage=Stage.FINAL_EDITORIAL_REPAIR,
            status=StageStatus.FAILED,
            artifacts=(
                "paper_editorial_input.md",
                "codex_review.json",
                "editorial_review.json",
                "editorial_iterations.json",
                "editorial_final_assessment.json",
            ),
            evidence_refs=(
                "stage-24/paper_editorial_input.md",
                "stage-24/codex_review.json",
                "stage-24/editorial_review.json",
                "stage-24/editorial_iterations.json",
                "stage-24/editorial_final_assessment.json",
            ),
            error="Local codex CLI is not available for Stage 24 editorial repair",
        )

    loop_result = _run_codex_editorial_loop(stage_dir, run_dir, source_markdown, config)
    table_caption_sources = tuple(
        path.read_text(encoding="utf-8")
        for path in (
            run_dir / "stage-24" / "paper_repaired.md",
            run_dir / "stage-23" / "paper_final_verified.md",
            run_dir / "stage-22" / "paper_final.md",
        )
        if path.exists() and path.stat().st_size > 0
    )
    normalized_markdown = _normalize_final_paper_markdown(
        loop_result.markdown,
        topic=config.research.topic,
        domains=config.research.domains,
        submission_profile=config.export.submission_profile,
        table_caption_sources=table_caption_sources,
        figure_reference_sources=table_caption_sources,
    )
    limited_markdown = _enforce_reference_limit(
        normalized_markdown,
        max_references=config.export.max_references,
    )
    repaired_markdown = limited_markdown
    if not _preserves_required_structure(normalized_markdown, limited_markdown):
        logger.warning(
            "Stage 24: reference-limit pass damaged markdown structure; keeping normalized markdown"
        )
        repaired_markdown = normalized_markdown
    (stage_dir / "paper_repaired.md").write_text(repaired_markdown, encoding="utf-8")
    _write_stage24_failure(
        stage_dir,
        review=loop_result.review,
        iterations=loop_result.iterations,
        assessment=loop_result.assessment,
    )

    _stage_editorial_compile_inputs(stage_dir, run_dir)
    _sync_bibliography_with_markdown(stage_dir, repaired_markdown)

    artifacts = [
        "paper_editorial_input.md",
        "paper_repaired.md",
        "codex_review.json",
        "editorial_review.json",
        "editorial_iterations.json",
        "editorial_final_assessment.json",
    ]
    evidence_refs = [
        "stage-24/paper_editorial_input.md",
        "stage-24/paper_repaired.md",
        "stage-24/codex_review.json",
        "stage-24/editorial_review.json",
        "stage-24/editorial_iterations.json",
        "stage-24/editorial_final_assessment.json",
    ]
    compile_artifacts, compile_evidence, _ = _compile_editorial_tex(
        stage_dir,
        repaired_markdown,
        config,
    )
    artifacts.extend(compile_artifacts)
    evidence_refs.extend(compile_evidence)
    docx_artifacts, docx_evidence, _ = _export_editorial_docx(
        stage_dir,
        authors=config.export.authors,
        bibliography_name=config.export.bib_file + ".bib",
    )
    _extend_unique(artifacts, docx_artifacts)
    _extend_unique(evidence_refs, docx_evidence)
    docx_quality_path = stage_dir / "docx_quality.json"
    docx_quality_payload = _load_docx_quality_payload(docx_quality_path)
    canonical_markdown_compressed_for_page_limit = False
    canonical_page_compression_rounds = 0
    compression_limit = max(1, config.experiment.editorial_repair.max_iterations)
    for compression_round in range(1, compression_limit + 1):
        constraint_quality = _audit_editorial_constraints(
            stage_dir,
            markdown=repaired_markdown,
            config=config,
        )
        existing_issues = (
            list(cast(list[object], docx_quality_payload.get("issues", [])))
            if isinstance(docx_quality_payload.get("issues", []), list)
            else []
        )
        existing_warnings = (
            list(cast(list[object], docx_quality_payload.get("warnings", [])))
            if isinstance(docx_quality_payload.get("warnings", []), list)
            else []
        )
        limit_issues = list(cast(list[object], constraint_quality.get("issues", [])))
        limit_warnings = list(cast(list[object], constraint_quality.get("warnings", [])))
        docx_quality_payload.update(constraint_quality)
        docx_quality_payload["issues"] = existing_issues + [
            issue for issue in limit_issues if issue not in existing_issues
        ]
        docx_quality_payload["warnings"] = existing_warnings + [
            warning for warning in limit_warnings if warning not in existing_warnings
        ]

        needs_page_compression = (
            config.export.docx_page_limit > 0
            and not docx_quality_payload.get("docx_page_limit_ok", True)
            and int(docx_quality_payload.get("docx_page_count", 0)) > 0
        )
        if not needs_page_compression:
            break

        compressed_markdown = _compress_markdown_for_docx_limit(
            repaired_markdown,
            current_page_count=int(docx_quality_payload.get("docx_page_count", 0)),
            page_limit=config.export.docx_page_limit,
            submission_profile=config.export.submission_profile,
            compression_round=compression_round,
        )
        if compressed_markdown.strip() == repaired_markdown.strip():
            break

        repaired_markdown = compressed_markdown
        canonical_markdown_compressed_for_page_limit = True
        canonical_page_compression_rounds = compression_round
        (stage_dir / "paper_repaired.md").write_text(
            repaired_markdown,
            encoding="utf-8",
        )
        _sync_bibliography_with_markdown(stage_dir, repaired_markdown)
        compile_artifacts, compile_evidence, _ = _compile_editorial_tex(
            stage_dir,
            repaired_markdown,
            config,
        )
        _extend_unique(artifacts, compile_artifacts)
        _extend_unique(evidence_refs, compile_evidence)
        docx_artifacts, docx_evidence, _ = _export_editorial_docx(
            stage_dir,
            authors=config.export.authors,
            bibliography_name=config.export.bib_file + ".bib",
        )
        _extend_unique(artifacts, docx_artifacts)
        _extend_unique(evidence_refs, docx_evidence)
        docx_quality_payload = _load_docx_quality_payload(docx_quality_path)

    docx_quality_path.write_text(
        json.dumps(docx_quality_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    assessment_path = stage_dir / "editorial_final_assessment.json"
    if assessment_path.exists():
        assessment_payload = json.loads(assessment_path.read_text(encoding="utf-8"))
        docx_quality = (
            json.loads(docx_quality_path.read_text(encoding="utf-8"))
            if docx_quality_path.exists()
            else {
                "clean": False,
                "heading_numbering_ok": False,
                "equation_alignment_ok": False,
                "display_math_omml_ok": False,
                "figure_caption_numbering_ok": False,
                "table_caption_numbering_ok": False,
                "issues": ["docx_not_exported"],
            }
        )
        assessment_payload["docx_clean"] = bool(docx_quality.get("clean", False))
        assessment_payload["docx_heading_numbering_ok"] = bool(
            docx_quality.get("heading_numbering_ok", False)
        )
        assessment_payload["docx_equation_alignment_ok"] = bool(
            docx_quality.get("equation_alignment_ok", False)
        )
        assessment_payload["docx_display_math_omml_ok"] = bool(
            docx_quality.get("display_math_omml_ok", False)
        )
        assessment_payload["docx_keywords_present"] = bool(
            docx_quality.get("keywords_present", False)
        )
        assessment_payload["docx_figure_caption_numbering_ok"] = bool(
            docx_quality.get("figure_caption_numbering_ok", False)
        )
        assessment_payload["docx_table_caption_numbering_ok"] = bool(
            docx_quality.get("table_caption_numbering_ok", False)
        )
        assessment_payload["docx_numeric_citations_plain"] = bool(
            docx_quality.get("numeric_citations_plain", False)
        )
        assessment_payload["reference_limit_ok"] = bool(
            docx_quality.get("reference_limit_ok", True)
        )
        assessment_payload["reference_count"] = int(docx_quality.get("reference_count", 0))
        assessment_payload["docx_page_limit_ok"] = bool(
            docx_quality.get("docx_page_limit_ok", True)
        )
        assessment_payload["docx_page_count"] = int(docx_quality.get("docx_page_count", 0))
        assessment_payload["docx_remaining_issues"] = list(
            cast(list[object], docx_quality.get("issues", []))
        ) if isinstance(docx_quality.get("issues", []), list) else []
        assessment_payload["pdf_docx_shared_canonical_content"] = True
        assessment_payload["canonical_markdown_source"] = "stage-24/paper_repaired.md"
        assessment_payload["canonical_markdown_compressed_for_page_limit"] = (
            canonical_markdown_compressed_for_page_limit
        )
        assessment_payload["canonical_page_compression_rounds"] = (
            canonical_page_compression_rounds
        )
        assessment_payload["docx_used_citeproc"] = bool(_pandoc_docx_citeproc_args(which("pandoc") or ""))
        assessment_path.write_text(
            json.dumps(assessment_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return StageResult(
        stage=Stage.FINAL_EDITORIAL_REPAIR,
        status=StageStatus.DONE
        if loop_result.success and not docx_quality_payload["issues"]
        else StageStatus.FAILED,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(evidence_refs),
        error=None
        if loop_result.success and not docx_quality_payload["issues"]
        else (loop_result.error or "; ".join(str(issue) for issue in docx_quality_payload["issues"])),
    )
