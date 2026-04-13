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
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_CAPTION_RE = re.compile(
    r"^(?:\*\*|\*)?\s*Table\s+\d+[.:]\s+.*(?:\*\*|\*)?\s*$",
    re.IGNORECASE,
)
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

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_DOCX_NS = {"w": _W_NS, "m": _M_NS}


@dataclass
class _Bundle:
    image_path: str
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
        return "<sup>[" + ", ".join(nums) + "]</sup>"

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
            style = "FirstParagraph" if idx == 0 else "BodyText"
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


def _postprocess_editorial_docx(docx_path: Path) -> dict[str, object]:
    quality: dict[str, object] = {
        "clean": False,
        "heading_numbering_ok": False,
        "removed_empty_paragraphs": 0,
        "styled_tables": 0,
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
        _scale_captioned_figures(document_root)
        _apply_docx_page_layout(document_root)
        _normalize_docx_styles(styles_root)
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
                "clean": True,
                "heading_numbering_ok": heading_numbering_ok,
                "removed_empty_paragraphs": removed,
                "styled_tables": styled_tables,
                "issues": [],
            }
        )
        return quality
    except Exception as exc:  # noqa: BLE001
        quality["issues"] = [f"postprocess_failed:{exc}"]
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
    image_name = Path(image_path).name
    if image_name.startswith("pipeline_overview_") and "protocol" in lowered:
        return True
    if image_name.startswith("architecture_diagram_") and (
        "architecture" in lowered or "model" in lowered
    ):
        return True
    caption_text = _clean_caption_text(caption_block or "").lower()
    if caption_text:
        keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", caption_text)[:6] if w not in {"figure", "across", "methods"}]
        overlap = sum(1 for word in keywords if word in lowered)
        if overlap >= 2:
            return True
    return any(cue in lowered for cue in ("shown", "below", "summar", "illustrat", "visual"))


def _extract_bundles(blocks: list[str]) -> list[_Bundle]:
    sections = _section_contexts(blocks)
    bundles: list[_Bundle] = []
    for idx, block in enumerate(blocks):
        image_match = _IMAGE_RE.search(block)
        if not image_match:
            continue
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
            caption_block,
        ):
            explanation_indices.append(trailing_idx)
            end = trailing_idx
        bundles.append(
            _Bundle(
                image_path=image_match.group(1),
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


def _find_first_figure_reference_index(
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
    text = re.sub(r"^\*\*Figure\s+\d+[.:]?\s*\*\*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\*Figure\s+\d+[.:]?\s*", "", text, flags=re.IGNORECASE)
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
    caption_block: str | None,
    section: str,
) -> str:
    image_name = Path(image_path).name
    if image_name.startswith("pipeline_overview_"):
        return "The end-to-end evaluation protocol is summarized visually below."
    if image_name.startswith("architecture_diagram_"):
        return "The model architecture is summarized visually below."

    caption_text = _clean_caption_text(caption_block or "")
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
        explanation = _build_explanation(bundle.image_path, caption_block, bundle.section)
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
            ref_idx = _find_first_figure_reference_index(blocks, bundle.figure_number)
            if ref_idx is not None and bundle.start - ref_idx > 2:
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
) -> tuple[list[str], list[str], bool]:
    artifacts: list[str] = []
    evidence: list[str] = []
    compile_ok = False
    try:
        from researchclaw.templates import get_template, markdown_to_latex
        from researchclaw.templates.compiler import compile_latex

        tpl = get_template(config.export.target_conference)
        tex = markdown_to_latex(
            repaired_markdown,
            tpl,
            title=_extract_title(repaired_markdown),
            authors=config.export.authors,
            bib_file=config.export.bib_file,
        )
        tex_path = stage_dir / "paper_repaired.tex"
        tex_path.write_text(tex, encoding="utf-8")
        artifacts.append("paper_repaired.tex")
        evidence.append("stage-24/paper_repaired.tex")

        compile_result = compile_latex(tex_path, max_attempts=2, timeout=120)
        if compile_result.success:
            compile_ok = True
            pdf_path = tex_path.with_suffix(".pdf")
            if pdf_path.exists():
                artifacts.append(pdf_path.name)
                evidence.append(f"stage-24/{pdf_path.name}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 24: Editorial LaTeX generation skipped: %s", exc)
    return artifacts, evidence, compile_ok


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
    return (
        f"# Stage 24 Editorial Repair Task\n\n"
        f"Iteration: {iteration}\n"
        f"Source: {source_label}\n\n"
        "You are repairing a research paper markdown draft. Edit `paper_repaired.md` in place.\n\n"
        f"Mode:\n{mode_instruction}\n\n"
        "Goals:\n"
        "1. Fix figure placement and figure-text proximity issues.\n"
        "2. Ensure every retained figure has a local explanation or explicit discussion.\n"
        "3. Improve local flow around figures and clean obvious editorial rough edges.\n"
        "4. Fix ugly layout outcomes when they make the paper look unfinished, including single-figure pages, awkward page breaks, large blank areas around floats, and figures or tables that visibly break the opening of the next section.\n"
        "5. Write a structured `codex_review.json` describing remaining issues, risks, and whether another round is needed.\n\n"
        "Hard constraints:\n"
        "- Do not change experiment numbers, metric values, or table values.\n"
        "- Do not add or remove citation keys.\n"
        "- Do not invent new experiments or change conclusions' factual meaning.\n"
        "- Do not rewrite unrelated sections.\n"
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
        current_markdown = payload
        issue_report = _audit_markdown(_split_blocks(current_markdown))
        compile_artifacts, _, compile_clean = _compile_editorial_tex(stage_dir, current_markdown, config)
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
    repaired_markdown = loop_result.markdown
    (stage_dir / "paper_repaired.md").write_text(repaired_markdown, encoding="utf-8")
    _write_stage24_failure(
        stage_dir,
        review=loop_result.review,
        iterations=loop_result.iterations,
        assessment=loop_result.assessment,
    )

    _stage_editorial_compile_inputs(stage_dir, run_dir)

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
        stage_dir, repaired_markdown, config
    )
    artifacts.extend(compile_artifacts)
    evidence_refs.extend(compile_evidence)
    docx_artifacts, docx_evidence, _ = _export_editorial_docx(
        stage_dir,
        authors=config.export.authors,
        bibliography_name=config.export.bib_file + ".bib",
    )
    artifacts.extend(docx_artifacts)
    evidence_refs.extend(docx_evidence)
    assessment_path = stage_dir / "editorial_final_assessment.json"
    if assessment_path.exists():
        assessment_payload = json.loads(assessment_path.read_text(encoding="utf-8"))
        docx_quality_path = stage_dir / "docx_quality.json"
        docx_quality = (
            json.loads(docx_quality_path.read_text(encoding="utf-8"))
            if docx_quality_path.exists()
            else {"clean": False, "heading_numbering_ok": False, "issues": ["docx_not_exported"]}
        )
        assessment_payload["docx_clean"] = bool(docx_quality.get("clean", False))
        assessment_payload["docx_heading_numbering_ok"] = bool(
            docx_quality.get("heading_numbering_ok", False)
        )
        assessment_payload["docx_remaining_issues"] = list(
            cast(list[object], docx_quality.get("issues", []))
        ) if isinstance(docx_quality.get("issues", []), list) else []
        assessment_payload["docx_used_citeproc"] = bool(_pandoc_docx_citeproc_args(which("pandoc") or ""))
        assessment_path.write_text(
            json.dumps(assessment_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return StageResult(
        stage=Stage.FINAL_EDITORIAL_REPAIR,
        status=StageStatus.DONE if loop_result.success else StageStatus.FAILED,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(evidence_refs),
        error=None if loop_result.success else loop_result.error,
    )
