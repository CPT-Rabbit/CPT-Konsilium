from __future__ import annotations

import logging
import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from .deid import (
    DeidentifiedDocument,
    PiiDetector,
    assert_no_blocking_residue,
    deidentify,
    institutional_email_allowlist,
    residue_report,
)
from .util import json_block

_LOG = logging.getLogger(__name__)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LAB = re.compile(r"\b(HbA1c|LDL|HDL|CRP|TSH|glucose)\b", re.I)
_MED = re.compile(r"\b(metformin|atorvastatin|insulin|levothyroxine|amlodipine)\b|\b\d+\s*mg\b", re.I)
_DOCUMENT_DATE = re.compile(r"\b(?:den|vom)\s+(\d{1,2}\.\d{1,2}\.\d{4})\b", re.I)
_METADATA_TOKEN = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")
_PERSON_TITLE = re.compile(r"\b(?:Dr\.?|Prof\.?|Frau|Herr)\b", re.I)
_MIN_PAGE_TEXT_CHARS = 12


Extractor = Callable[[Path], str]


@dataclass(frozen=True)
class ExtractedText:
    text: str
    stats: dict


def ingest_document(
    patient_id: str,
    path: str | Path,
    root: str | Path,
    *,
    extractor: Extractor | None = None,
    pii_detector: PiiDetector | None = None,
    structure_model=None,
    allow_synthetic: bool = False,
    residue_policy: dict[str, str] | None = None,
) -> Path:
    path = Path(path)
    text = extract_text_with_stats(path, extractor=extractor).text
    return ingest_text(
        patient_id,
        text,
        root,
        pii_detector=pii_detector,
        structure_model=structure_model,
        allow_synthetic=allow_synthetic,
        residue_policy=residue_policy,
    )


def ingest_text(
    patient_id: str,
    text: str,
    root: str | Path,
    *,
    pii_detector: PiiDetector | None = None,
    structure_model=None,
    allow_synthetic: bool = False,
    residue_policy: dict[str, str] | None = None,
) -> Path:
    if pii_detector is None and not allow_synthetic:
        raise RuntimeError("ingest requires a configured PII detector; pass allow_synthetic=True only for tests")
    if structure_model is None and not allow_synthetic:
        raise RuntimeError("ingest requires a configured structuring model; pass allow_synthetic=True only for tests")
    root = Path(root)
    safe_patient_id = _safe_id(patient_id)
    vault_path = root / "identity_vault" / f"{safe_patient_id}.json"
    existing_vault = _read_vault(vault_path)
    document = deidentify(text, existing_vault=existing_vault, pii_detector=pii_detector)
    return _store_deidentified(
        safe_patient_id,
        document,
        root,
        structure_model=structure_model,
        residue_policy=residue_policy,
    )


def _store_deidentified(
    patient_id: str,
    document: DeidentifiedDocument,
    root: Path,
    *,
    structure_model=None,
    residue_policy: dict[str, str] | None = None,
) -> Path:
    assert_no_blocking_residue(
        document.text,
        residue_policy,
        retained_institutional_emails=document.retained_institutional_emails,
    )
    structured = _structure_document(document.text, structure_model=structure_model)
    metadata = _document_metadata(document.text, structured)
    patient_dir = root / "patients" / patient_id
    documents_dir = patient_dir / "documents"
    vault_dir = root / "identity_vault"
    vault_path = vault_dir / f"{patient_id}.json"
    documents_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)
    document_path = _next_document_path(documents_dir, metadata)
    _write_document(document_path, document.text, metadata)
    _write_structured_files(patient_dir, structured)
    (patient_dir / "passport.md").write_text(
        f"# Patient {patient_id}\n\nDe-identified source documents are in `documents/`.\n",
        encoding="utf-8",
    )
    vault_path.write_text(
        json.dumps(document.vault, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _sync_memory(root)
    return document_path


def ingest_from_preview(
    config,
    patient_id: str,
    preview_path: str | Path,
    root: str | Path | None = None,
    *,
    structure_model=None,
) -> Path:
    if not config.runtime.allow_real_patient_docs:
        raise RuntimeError("reviewed-preview ingest is disabled by runtime.allow_real_patient_docs")
    root = Path(root or config.runtime.patient_root)
    preview = Path(preview_path).resolve()
    previews_root = (root / "previews").resolve()
    if preview.parent != previews_root or not preview.name.startswith("preview-") or preview.suffix != ".md":
        raise ValueError("--from-preview requires memory/previews/preview-*.md")
    vault_path = preview.with_suffix(".vault.json")
    if not vault_path.exists():
        raise RuntimeError(f"reviewed preview vault is missing: {vault_path.name}")
    text = preview.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError("reviewed preview is empty")
    preview_vault = _read_vault(vault_path)
    if not isinstance(preview_vault, dict) or any(
        not re.fullmatch(r"\[[A-Z][A-Z_]*_\d+\]", key) or not isinstance(value, str)
        for key, value in preview_vault.items()
    ):
        raise RuntimeError("reviewed preview vault has invalid token mappings")
    safe_patient_id = _safe_id(patient_id)
    existing_vault = _read_vault(root / "identity_vault" / f"{safe_patient_id}.json")
    conflicts = [key for key, value in preview_vault.items() if key in existing_vault and existing_vault[key] != value]
    if conflicts:
        raise RuntimeError(f"reviewed preview vault conflicts with existing patient tokens: {', '.join(conflicts)}")
    document = DeidentifiedDocument(
        text=text,
        vault={**existing_vault, **preview_vault},
        retained_institutional_emails=institutional_email_allowlist(text),
    )
    return _store_deidentified(
        safe_patient_id,
        document,
        root,
        structure_model=structure_model or _structure_model(config),
        residue_policy=config.deidentification.residue,
    )


def ingest_patient_document(
    config,
    patient_id: str,
    text: str,
    root: str | Path | None = None,
    *,
    pii_detector: PiiDetector | None = None,
    structure_model=None,
) -> Path:
    detector, model = _real_ingest_components(config, pii_detector=pii_detector, structure_model=structure_model)
    return ingest_text(
        patient_id,
        text,
        root or config.runtime.patient_root,
        pii_detector=detector,
        structure_model=model,
        residue_policy=config.deidentification.residue,
    )


def ingest_patient_file(
    config,
    patient_id: str,
    path: str | Path,
    root: str | Path | None = None,
    *,
    extractor: Extractor | None = None,
    pii_detector: PiiDetector | None = None,
    structure_model=None,
) -> tuple[Path, dict]:
    detector, model = _real_ingest_components(config, pii_detector=pii_detector, structure_model=structure_model)
    extracted = extract_text_with_stats(path, extractor=extractor)
    document_path = ingest_text(
        patient_id,
        extracted.text,
        root or config.runtime.patient_root,
        pii_detector=detector,
        structure_model=model,
        residue_policy=config.deidentification.residue,
    )
    return document_path, extracted.stats


def _real_ingest_components(config, *, pii_detector: PiiDetector | None = None, structure_model=None):
    if not config.runtime.allow_real_patient_docs:
        raise RuntimeError("real patient ingest is disabled by runtime.allow_real_patient_docs")
    if not config.deidentification.ollama_model:
        raise RuntimeError("real patient ingest requires deidentification.ollama_model")
    detector = pii_detector or _ollama_detector(config)
    _check_detector(detector)
    model = structure_model or _structure_model(config)
    return detector, model


def _ollama_detector(config):
    from .ollama_deid import OllamaPiiDetector

    return OllamaPiiDetector(
        model=config.deidentification.ollama_model,
        base_url=config.deidentification.ollama_url,
        timeout_s=config.deidentification.timeout_s,
        chunk_size=config.deidentification.chunk_size,
        chunk_overlap=config.deidentification.chunk_overlap,
    )


def deid_preview(
    config,
    path: str | Path,
    root: str | Path | None = None,
    *,
    extractor: Extractor | None = None,
    pii_detector: PiiDetector | None = None,
) -> dict:
    """Extract and de-identify locally without patient-memory or model-provider writes."""
    extracted = extract_text_with_stats(path, extractor=extractor)
    detector = pii_detector or (_ollama_detector(config) if config.deidentification.ollama_model else None)
    if detector is not None:
        _check_detector(detector)
    document = deidentify(extracted.text, pii_detector=detector)
    hits = residue_report(
        document.text,
        config.deidentification.residue,
        retained_institutional_emails=document.retained_institutional_emails,
    )
    preview_dir = Path(root or config.runtime.patient_root) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_id(Path(path).stem) if Path(path).stem.strip() else "document"
    preview_path = preview_dir / f"preview-{name}.md"
    vault_path = preview_dir / f"preview-{name}.vault.json"
    report_path = preview_dir / f"preview-{name}.residue.json"
    preview_path.write_text(document.text.strip() + "\n", encoding="utf-8")
    vault_path.write_text(json.dumps(document.vault, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "hits": [hit.__dict__ for hit in hits],
        "rejected_entities": [item.__dict__ for item in document.rejected_entities],
        "blocked": any(hit.action == "block" for hit in hits),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "preview_path": str(preview_path),
        "vault_path": str(vault_path),
        "residue_report_path": str(report_path),
        "residue": report,
        "extraction": extracted.stats,
    }


def _check_detector(detector: PiiDetector) -> None:
    try:
        detector("Detector reachability check.")
    except Exception as error:
        raise RuntimeError("PII detector is not reachable") from error


def _structure_model(config):
    from .model_client import ModelClient
    from .providers.base import build_provider

    return ModelClient(
        build_provider(config.model),
        request_timeout_s=config.model.request_timeout_s,
        stream=config.model.stream,
    )


def _extract_text(path: Path, *, extractor: Extractor | None = None) -> str:
    return extract_text_with_stats(path, extractor=extractor).text


def extract_text_with_stats(
    path: str | Path,
    *,
    extractor: Extractor | None = None,
    pdf_reader=None,
    ocr_pages=None,
) -> ExtractedText:
    path = Path(path)
    if extractor is not None:
        text = extractor(path)
        if not text.strip():
            raise RuntimeError(f"extracted document is empty: {path}")
        stats = {"kind": "override", "pages_total": None, "text_layer_pages": [], "ocr_pages": [], "suspicious_pages": []}
        _LOG.info("document extraction stats: %s", stats)
        return ExtractedText(text, stats)
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise RuntimeError(f"extracted document is empty: {path}")
        stats = {"kind": "text", "pages_total": 1, "text_layer_pages": [1], "ocr_pages": [], "suspicious_pages": []}
        _LOG.info("document extraction stats: %s", stats)
        return ExtractedText(text, stats)
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path, pdf_reader=pdf_reader, ocr_pages=ocr_pages)
    raise ValueError(f"unsupported document type: {path.suffix or path.name}")


def _extract_pdf_text(path: Path, *, pdf_reader=None, ocr_pages=None) -> ExtractedText:
    page_texts = _read_pdf_text_pages(path, pdf_reader=pdf_reader)
    if not page_texts:
        raise RuntimeError(f"PDF has no pages: {path}")
    weak_pages = [index for index, text in enumerate(page_texts, 1) if _weak_text(text)]
    text_layer_pages = [index for index in range(1, len(page_texts) + 1) if index not in weak_pages]
    if weak_pages:
        ocr_texts = (ocr_pages or _ocr_pdf_pages)(path, weak_pages)
        for page in weak_pages:
            page_texts[page - 1] = ocr_texts.get(page, "")
    empty_pages = [index for index, text in enumerate(page_texts, 1) if not text.strip()]
    if empty_pages:
        pages = ", ".join(str(page) for page in empty_pages)
        raise RuntimeError(f"PDF pages without text after OCR: {pages}")
    suspicious_pages = [index for index, text in enumerate(page_texts, 1) if _weak_text(text)]
    stats = {
        "kind": "pdf",
        "pages_total": len(page_texts),
        "text_layer_pages": text_layer_pages,
        "ocr_pages": weak_pages,
        "suspicious_pages": suspicious_pages,
    }
    _LOG.info("document extraction stats: %s", stats)
    return ExtractedText("\n\n".join(text.strip() for text in page_texts), stats)


def _read_pdf_text_pages(path: Path, *, pdf_reader=None) -> list[str]:
    if pdf_reader is not None:
        return [page.extract_text() or "" for page in pdf_reader(path)]
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as error:
        raise RuntimeError("PDF extraction requires pypdf or an explicit extractor") from error
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _ocr_pdf_pages(path: Path, pages: list[int]) -> dict[int, str]:
    if shutil.which("ocrmypdf") is None:
        raise RuntimeError(
            "OCR required but not available: install ocrmypdf and tesseract language packs deu+eng"
        )
    with TemporaryDirectory() as tmp:
        output = Path(tmp) / "ocr.pdf"
        command = [
            "ocrmypdf",
            "--force-ocr",
            "--deskew",
            "--rotate-pages",
            "--output-type",
            "pdf",
            "--pages",
            ",".join(str(page) for page in pages),
            "-l",
            "deu+eng",
            str(path),
            str(output),
        ]
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        if run.returncode:
            detail = (run.stderr or run.stdout).strip().splitlines()[-1:]
            raise RuntimeError(f"OCR failed: {detail[0] if detail else run.returncode}")
        page_texts = _read_pdf_text_pages(output)
    return {page: page_texts[page - 1] if page <= len(page_texts) else "" for page in pages}


def _weak_text(text: str) -> bool:
    return len(re.sub(r"\s+", "", text or "")) < _MIN_PAGE_TEXT_CHARS


def _read_vault(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _structure_document(text: str, *, structure_model=None) -> dict:
    structured = _prepass_structure(text)
    if structure_model is not None:
        structured = _merge_structure(structured, _model_structure(text, structured, structure_model))
    return structured


def _write_structured_files(patient_dir: Path, structured: dict) -> None:
    (patient_dir / "timeline").mkdir(exist_ok=True)
    (patient_dir / "labs").mkdir(exist_ok=True)
    for path, title, key in (
        (patient_dir / "timeline" / "events.md", "Timeline", "timeline"),
        (patient_dir / "problems.md", "Problems", "problems"),
        (patient_dir / "meds.md", "Medications", "meds"),
        (patient_dir / "labs" / "labs.md", "Labs", "labs"),
    ):
        _write_md(path, title, _dedupe([*_read_md_items(path), *structured[key]]))
    _write_md(patient_dir / "strategy.md", "Strategy", ["Review changes with a human clinician."])


def _prepass_structure(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "timeline": [line for line in lines if _DATE.search(line)],
        "problems": [_after_label(line, "Problem") for line in lines if line.lower().startswith("problem:")],
        "meds": [line for line in lines if _MED.search(line)],
        "labs": [line for line in lines if _LAB.search(line)],
        "document_date": _body_document_date(text) or "",
        "document_topic": "",
        "document_sender": "",
    }


def _model_structure(text: str, prepass: dict, structure_model) -> dict:
    system_prompt = (
        "Extract clinical structure from de-identified medical text. "
        "Return JSON object with array fields timeline, problems, meds, labs and string fields "
        "document_date, document_topic, document_sender. document_date is the letter's own date "
        "as YYYY-MM-DD; document_topic is a short subject; document_sender is the issuing institution. "
        "Topic and sender must never be a person. "
        "Do not infer PII and do not restore token values."
    )
    messages = [
        {
            "role": "user",
            "content": json.dumps({"text": text, "prepass": prepass}, ensure_ascii=False),
        }
    ]
    if hasattr(structure_model, "build_kwargs") and hasattr(structure_model, "call"):
        response = structure_model.call(structure_model.build_kwargs(messages, system_prompt, [], json_mode=True))
        payload = getattr(response, "content", response)
    else:
        payload = structure_model(messages, system_prompt)
    data = json.loads(json_block(payload)) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("structuring model response must be a JSON object")
    return _coerce_structure(data)


def _coerce_structure(data: dict) -> dict:
    out = {}
    for key in ("timeline", "problems", "meds", "labs"):
        values = data.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"structuring model field must be a list: {key}")
        out[key] = [str(item).strip() for item in values if str(item).strip()]
    for key in ("document_date", "document_topic", "document_sender"):
        value = data.get(key, "")
        out[key] = value.strip() if isinstance(value, str) else ""
    return out


def _merge_structure(first: dict, second: dict) -> dict:
    merged = {
        key: _dedupe([*first.get(key, []), *second.get(key, [])])
        for key in ("timeline", "problems", "meds", "labs")
    }
    for key in ("document_date", "document_topic", "document_sender"):
        merged[key] = second.get(key) or first.get(key) or ""
    return merged


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _write_md(path: Path, title: str, items: list[str]) -> None:
    body = "\n".join(f"- {item}" for item in items) if items else "- No structured entries yet."
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _read_md_items(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line[2:]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ") and line != "- No structured entries yet."
    ]


def _document_metadata(text: str, structured: dict) -> dict[str, str]:
    document_date = _iso_date(structured.get("document_date")) or _body_document_date(text)
    date_source = "document" if document_date else "ingest"
    document_date = document_date or date.today().isoformat()
    return {
        "date": document_date,
        "date_source": date_source,
        "topic": _metadata_slug(structured.get("document_topic"), "document"),
        "sender": _metadata_slug(structured.get("document_sender"), "unknown-sender"),
    }


def _metadata_slug(value, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    candidate = value.strip()
    if _METADATA_TOKEN.search(candidate) or _PERSON_TITLE.search(candidate) or residue_report(candidate):
        return fallback
    candidate = candidate.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}))
    candidate = unicodedata.normalize("NFKD", candidate).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", candidate).strip("-")
    return slug or fallback


def _next_document_path(documents_dir: Path, metadata: dict[str, str]) -> Path:
    base = f"{metadata['date']}_{metadata['topic']}_{metadata['sender']}"
    assert_no_blocking_residue(base)
    path = documents_dir / f"{base}.md"
    suffix = 2
    while path.exists():
        path = documents_dir / f"{base}-{suffix}.md"
        suffix += 1
    return path


def _write_document(path: Path, text: str, metadata: dict[str, str]) -> None:
    frontmatter = (
        "---\n"
        f"document_date: {metadata['date']}\n"
        f"date_source: {metadata['date_source']}\n"
        f"topic: {metadata['topic']}\n"
        f"sender: {metadata['sender']}\n"
        "---\n\n"
    )
    path.write_text(frontmatter + text.strip() + "\n", encoding="utf-8")


def _body_document_date(text: str) -> str | None:
    match = _DOCUMENT_DATE.search(text)
    if not match:
        return None
    day, month, year = map(int, match.group(1).split("."))
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _iso_date(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def _after_label(line: str, label: str) -> str:
    return line.split(":", 1)[1].strip() if ":" in line else line.removeprefix(label).strip()


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not clean:
        raise ValueError("patient_id is empty after normalization")
    return clean


def _sync_memory(root: Path) -> None:
    from .memory import PatientMemory

    PatientMemory(root).sync()
