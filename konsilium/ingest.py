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

from .declutter import declutter, declutter_stats
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
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b")
_LAB = re.compile(r"\b(HbA1c|LDL|HDL|CRP|TSH|glucose)\b", re.I)
_MED = re.compile(r"\b(metformin|atorvastatin|insulin|levothyroxine|amlodipine)\b|\b\d+\s*mg\b", re.I)
_DOCUMENT_DATE = re.compile(r"\b(?:den|vom)\s+(\d{1,2}\.\d{1,2}\.\d{4})\b", re.I)
_METADATA_TOKEN = re.compile(r"\[(?:\d+_)?[A-Z][A-Z_]*_\d+\]")
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
    document = deidentify(
        text,
        existing_vault=existing_vault,
        pii_detector=pii_detector,
        known_identity=existing_vault or None,
    )
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
    fallback_date: str | None = None,
    accepted_residue: list[str] | None = None,
) -> Path:
    # Strip institutional boilerplate before the gate and any model egress, so
    # the stored/indexed body is clinical content only. Provenance (date, type,
    # sender) is preserved separately in metadata below.
    body = declutter(document.text)
    assert_no_blocking_residue(
        body,
        residue_policy,
        retained_institutional_emails=document.retained_institutional_emails,
    )
    structured = _structure_document(body, structure_model=structure_model)
    metadata = _document_metadata(body, structured)
    if metadata["date_source"] == "ingest" and fallback_date:
        metadata.update(date=fallback_date, date_source="filename")
    patient_dir = root / "patients" / patient_id
    documents_dir = patient_dir / "documents"
    vault_dir = root / "identity_vault"
    vault_path = vault_dir / f"{patient_id}.json"
    documents_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)
    document_path = _next_document_path(documents_dir, metadata)
    rendered = _render_document(body, metadata, structured, accepted_residue=accepted_residue)
    # Model-written sections pass the same gate as the body before touching disk.
    # The model may normalize OCR-garbled institutional emails, so the allowlist
    # is recomputed on the render too (generic institutional local-parts only).
    assert_no_blocking_residue(
        rendered,
        residue_policy,
        retained_institutional_emails=tuple(sorted({
            *document.retained_institutional_emails,
            *institutional_email_allowlist(rendered),
        })),
    )
    document_path.write_text(rendered, encoding="utf-8")
    structured["timeline"] = [f"{item} [doc:{document_path.stem}]" for item in structured["timeline"]]
    _write_structured_files(patient_dir, structured)
    (patient_dir / "passport.md").write_text(
        f"# Patient {patient_id}\n\nDe-identified source documents are in `documents/`.\n",
        encoding="utf-8",
    )
    vault_path.write_text(
        json.dumps(document.vault, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_token_ledger(vault_dir, patient_id, document.vault)
    _sync_memory(root)
    return document_path


_LEDGER_KIND_ORDER = ("PATIENT", "ADDR", "DOB", "PHONE", "EMAIL", "INSURANCE", "CASE_NUMBER")


def write_token_ledger(vault_dir: Path, patient_id: str, vault: dict[str, str]) -> Path:
    """Operator-readable token->cleartext register beside the JSON vault. Lives in
    identity_vault/ (never indexed, never in agent context); one file per patient
    so overlapping doctors/clinics stay isolated between patients. The letter
    renderer and manual review both read from this single central list."""
    from .deid import _token_kind

    groups: dict[str, list[tuple[int, str, str]]] = {}
    for token, value in vault.items():
        match = re.fullmatch(r"\[(?:\d+_)?[A-Z][A-Z_]*_(\d+)\]", token)
        index = int(match.group(1)) if match else 0
        groups.setdefault(_token_kind(token), []).append((index, token, value))
    lines = [
        f"# Token-Register — {patient_id}",
        "",
        f"Token → Klartext für alle Dokumente dieses Patienten ({len(vault)} Tokens).",
        "Nur lokal, außerhalb der indizierten Akte. Nach jedem Ingest aktualisiert.",
    ]
    ordered_kinds = [*_LEDGER_KIND_ORDER, *sorted(set(groups) - set(_LEDGER_KIND_ORDER))]
    for kind in ordered_kinds:
        entries = groups.get(kind)
        if not entries:
            continue
        lines += ["", f"## {kind}", "", "| Token | Klartext |", "| --- | --- |"]
        for _, token, value in sorted(entries):
            lines.append(f"| `{token}` | {value.replace('|', '\\|')} |")
    path = vault_dir / f"{patient_id}.tokens.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ingest_from_preview(
    config,
    patient_id: str,
    preview_path: str | Path,
    root: str | Path | None = None,
    *,
    structure_model=None,
    accepted_residue: list[str] | None = None,
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
        not re.fullmatch(r"\[(?:\d+_)?[A-Z][A-Z_]*_\d+\]", key) or not isinstance(value, str)
        for key, value in preview_vault.items()
    ):
        raise RuntimeError("reviewed preview vault has invalid token mappings")
    safe_patient_id = _safe_id(patient_id)
    existing_vault = _read_vault(root / "identity_vault" / f"{safe_patient_id}.json")
    preview_vault, text = _reconcile_preview_tokens(
        existing_vault, preview_vault, text, _patient_scope(safe_patient_id)
    )
    conflicts = [key for key, value in preview_vault.items() if key in existing_vault and existing_vault[key] != value]
    if conflicts:
        raise RuntimeError(f"reviewed preview vault conflicts with existing patient tokens: {', '.join(conflicts)}")
    document = DeidentifiedDocument(
        text=text,
        vault={**existing_vault, **preview_vault},
        retained_institutional_emails=institutional_email_allowlist(text),
    )
    residue_policy = dict(config.deidentification.residue)
    if accepted_residue:
        # Operator-reviewed acceptance: the named patterns were inspected on this
        # preview and confirmed non-PII; they report instead of blocking for this
        # single ingest. Never configurable globally.
        residue_policy.update({str(name).upper(): "report" for name in accepted_residue})
    return _store_deidentified(
        safe_patient_id,
        document,
        root,
        structure_model=structure_model or _structure_model(config),
        residue_policy=residue_policy,
        fallback_date=_filename_date(preview.stem),
        accepted_residue=[str(name).upper() for name in accepted_residue or []],
    )


def _filename_date(stem: str) -> str | None:
    match = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})", stem)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


_INBOX_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}


def inbox_documents_to_preview(inbox_dir: Path, previews_dir: Path) -> list[Path]:
    """New inbox documents that have no preview yet, so a re-run only processes
    freshly dropped files (idempotent watch-folder pass)."""
    inbox_dir = Path(inbox_dir)
    if not inbox_dir.is_dir():
        return []
    todo = []
    for path in sorted(inbox_dir.iterdir()):
        if path.suffix.lower() not in _INBOX_SUFFIXES or not path.is_file():
            continue
        name = _safe_id(path.stem) if path.stem.strip() else "document"
        if not (previews_dir / f"preview-{name}.md").exists():
            todo.append(path)
    return todo


def _patient_scope(patient_id: str) -> str | None:
    # Tokens are scoped by the patient's number so they stay unique across the
    # whole base. Known limit: patient ids must contain a unique number
    # (patient-1, patient-2, ...); a non-numeric id uses unscoped tokens.
    match = re.search(r"\d+", patient_id)
    return match.group(0) if match else None


def _reconcile_preview_tokens(
    existing_vault: dict[str, str], preview_vault: dict[str, str], text: str, scope: str | None
) -> tuple[dict[str, str], str]:
    """Previews number tokens from scratch and unscoped; remap every token onto
    the patient's scoped vault so an identical value keeps one identity token
    across all of that patient's documents and new values take the next free
    scoped index ([1_PATIENT_5])."""
    from .deid import _counts, _token_kind

    def scoped(kind: str, index: int) -> str:
        return f"[{scope}_{kind}_{index}]" if scope else f"[{kind}_{index}]"

    by_value = {(_token_kind(token), value): token for token, value in existing_vault.items()}
    counts = _counts(existing_vault)
    mapping = {}
    for token, value in preview_vault.items():
        kind = _token_kind(token)
        if existing_vault.get(token) == value:
            continue
        known = by_value.get((kind, value))
        if known:
            mapping[token] = known
        else:
            counts[kind] = counts.get(kind, 0) + 1
            new_token = scoped(kind, counts[kind])
            by_value[(kind, value)] = new_token
            mapping[token] = new_token
    if not mapping:
        return preview_vault, text
    remapped = {mapping.get(token, token): value for token, value in preview_vault.items()}
    text = re.sub(r"\[[A-Z][A-Z_]*_\d+\]", lambda match: mapping.get(match.group(0), match.group(0)), text)
    return remapped, text


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
    if not (config.deidentification.ollama_model or config.deidentification.gliner_model):
        raise RuntimeError("real patient ingest requires a configured PII detector model")
    detector = pii_detector or _build_detector(config)
    _check_detector(detector)
    model = structure_model or _structure_model(config)
    return detector, model


def _build_detector(config):
    """GLiNER NER is the primary recall layer; the Ollama model is the second
    opinion for prose-embedded names. Entities are unioned (fail-closed: more
    recall, never less)."""
    settings = config.deidentification
    detectors = []
    if settings.gliner_model:
        from .gliner_deid import GlinerPiiDetector

        detectors.append(GlinerPiiDetector(
            model=settings.gliner_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ))
    if settings.ollama_model:
        from .ollama_deid import OllamaPiiDetector

        detectors.append(OllamaPiiDetector(
            model=settings.ollama_model,
            base_url=settings.ollama_url,
            timeout_s=settings.timeout_s,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ))
    if not detectors:
        return None
    if len(detectors) == 1:
        return detectors[0]
    from .gliner_deid import composite_detector

    return composite_detector(*detectors)


def deid_preview(
    config,
    path: str | Path,
    root: str | Path | None = None,
    *,
    extractor: Extractor | None = None,
    pii_detector: PiiDetector | None = None,
    known_identity: dict[str, str] | None = None,
) -> dict:
    """Extract and de-identify locally without patient-memory or model-provider writes."""
    extracted = extract_text_with_stats(path, extractor=extractor)
    detector = pii_detector or _build_detector(config)
    if detector is not None:
        _check_detector(detector)
    document = deidentify(extracted.text, pii_detector=detector, known_identity=known_identity)
    body = declutter(document.text)
    hits = residue_report(
        body,
        config.deidentification.residue,
        retained_institutional_emails=document.retained_institutional_emails,
    )
    preview_dir = Path(root or config.runtime.patient_root) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_id(Path(path).stem) if Path(path).stem.strip() else "document"
    preview_path = preview_dir / f"preview-{name}.md"
    vault_path = preview_dir / f"preview-{name}.vault.json"
    report_path = preview_dir / f"preview-{name}.residue.json"
    preview_path.write_text(body.strip() + "\n", encoding="utf-8")
    vault_path.write_text(json.dumps(document.vault, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "hits": [hit.__dict__ for hit in hits],
        "rejected_entities": [item.__dict__ for item in document.rejected_entities],
        "blocked": any(hit.action == "block" for hit in hits),
        "declutter": declutter_stats(document.text),
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
        for attempt in (1, 2):
            try:
                return _merge_structure(structured, _model_structure(text, structured, structure_model))
            except Exception as exc:
                # The prepass is deterministic and derived from the already-gated body,
                # so a persistent model/HTTP/JSON failure degrades instead of aborting ingest.
                _LOG.warning(
                    "structuring model failed (%s, attempt %d); %s",
                    exc.__class__.__name__, attempt,
                    "retrying" if attempt == 1 else "using deterministic prepass",
                )
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


_TEMPLATE_FIELDS = ("doc_type", "specialty", "examination_date", "institution", "examiner", "summary")
_SECTION_TITLES = (
    ("kontext", "Kontext"),
    ("untersuchung", "Untersuchung"),
    ("befund", "Befund"),
    ("beurteilung", "Beurteilung"),
    ("empfehlung", "Empfehlung / Procedere"),
)


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
        **{key: "" for key in _TEMPLATE_FIELDS},
        "sections": {},
    }


def _model_structure(text: str, prepass: dict, structure_model) -> dict:
    system_prompt = (
        "Extract clinical structure from de-identified medical text. "
        "Return JSON object with array fields timeline, problems, meds, labs; string fields "
        "document_date, document_topic, document_sender, doc_type, specialty, examination_date, "
        "institution, examiner, summary; and object field sections with string fields "
        "kontext, untersuchung, befund, beurteilung, empfehlung. "
        "document_date is the letter's own date as YYYY-MM-DD; examination_date is when the "
        "examination itself took place (YYYY-MM-DD, may differ from the letter date). "
        "document_topic is a short subject; document_sender and institution are the issuing "
        "institution. doc_type is one of: arztbrief, eeg_befund, mrt_befund, labor, gutachten, "
        "medikationsplan, therapiebericht, entlassungsbericht, sonstiges. specialty is the medical "
        "specialty in lowercase German. examiner is the responsible clinician exactly as tokenized, "
        "e.g. '[PATIENT_2], Fachärztin für Kinder- und Jugendmedizin' — never a raw name. "
        "summary is 1-3 German sentences: what was examined, by whom (role), when, and the key "
        "conclusion. timeline entries are 'YYYY-MM-DD: Ereignis' lines for every dated medical "
        "event in the text; meds list medications with doses; labs list measured findings. "
        "sections rewrite the document into clean readable German, fixing OCR artifacts "
        "and dropping administrative noise, preserving every [TOKEN_n] token verbatim and every "
        "medical fact; never invent content; use an empty string for absent sections. "
        "Sections must not contain contact blocks (phone, email, web addresses) and must never "
        "mention birth dates or markers like 'geb.' — refer to the patient's age only. "
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
    for key in ("document_date", "document_topic", "document_sender", *_TEMPLATE_FIELDS):
        value = data.get(key, "")
        out[key] = value.strip() if isinstance(value, str) else ""
    sections = data.get("sections", {})
    out["sections"] = {
        key: str(sections.get(key, "")).strip()
        for key, _ in _SECTION_TITLES
    } if isinstance(sections, dict) else {}
    return out


def _merge_structure(first: dict, second: dict) -> dict:
    merged = {
        key: _dedupe([*first.get(key, []), *second.get(key, [])])
        for key in ("timeline", "problems", "meds", "labs")
    }
    for key in ("document_date", "document_topic", "document_sender", *_TEMPLATE_FIELDS):
        merged[key] = second.get(key) or first.get(key) or ""
    merged["sections"] = second.get("sections") or first.get("sections") or {}
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


def _render_document(
    text: str, metadata: dict[str, str], structured: dict, accepted_residue: list[str] | None = None
) -> str:
    front = {
        "document_date": metadata["date"],
        "examination_date": _iso_date(structured.get("examination_date")) or metadata["date"],
        "date_source": metadata["date_source"],
        "doc_type": structured.get("doc_type") or "sonstiges",
        "specialty": structured.get("specialty") or "",
        "institution": structured.get("institution") or "",
        "examiner": structured.get("examiner") or "",
        "topic": metadata["topic"],
        "sender": metadata["sender"],
        "summary": structured.get("summary") or "",
    }
    if accepted_residue:
        front["accepted_residue"] = sorted(set(accepted_residue))
    lines = ["---"]
    lines += [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items()]
    lines += ["---", ""]
    title_institution = f" — {front['institution']}" if front["institution"] else ""
    lines.append(f"# {front['doc_type']} — {front['examination_date']}{title_institution}")
    sections = structured.get("sections") or {}
    for key, title in _SECTION_TITLES:
        content = sections.get(key, "").strip()
        if content:
            lines += ["", f"## {title}", "", content]
    lines += ["", "## Originaltext (bereinigt)", "", text.strip(), ""]
    return "\n".join(lines)


def _write_document(path: Path, text: str, metadata: dict[str, str], structured: dict) -> None:
    # This shared send boundary has no residue-acceptance path.
    path.write_text(_render_document(text, metadata, structured), encoding="utf-8")


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
    if not clean or set(clean) <= {"."}:
        # Reject empty and dots-only ("." / ".." / ...) — the latter would collapse
        # or traverse the per-patient directory boundary.
        raise ValueError("patient_id is empty or path-unsafe after normalization")
    return clean


def _sync_memory(root: Path) -> None:
    from .memory import PatientMemory

    PatientMemory(root).sync()
