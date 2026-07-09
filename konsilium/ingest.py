from __future__ import annotations

import logging
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from .deid import PiiDetector, deidentify

_LOG = logging.getLogger(__name__)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LAB = re.compile(r"\b(HbA1c|LDL|HDL|CRP|TSH|glucose)\b", re.I)
_MED = re.compile(r"\b(metformin|atorvastatin|insulin|levothyroxine|amlodipine)\b|\b\d+\s*mg\b", re.I)
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
    )


def ingest_text(
    patient_id: str,
    text: str,
    root: str | Path,
    *,
    pii_detector: PiiDetector | None = None,
    structure_model=None,
    allow_synthetic: bool = False,
) -> Path:
    if pii_detector is None and not allow_synthetic:
        raise RuntimeError("ingest requires a configured PII detector; pass allow_synthetic=True only for tests")
    if structure_model is None and not allow_synthetic:
        raise RuntimeError("ingest requires a configured structuring model; pass allow_synthetic=True only for tests")
    root = Path(root)
    safe_patient_id = _safe_id(patient_id)
    patient_dir = root / "patients" / safe_patient_id
    vault_dir = root / "identity_vault"
    patient_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)

    vault_path = vault_dir / f"{safe_patient_id}.json"
    existing_vault = _read_vault(vault_path)
    document = deidentify(text, existing_vault=existing_vault, pii_detector=pii_detector)
    documents_path = patient_dir / "documents.md"
    _append_document(documents_path, document.text)
    _write_structured_files(
        patient_dir,
        documents_path.read_text(encoding="utf-8"),
        structure_model=structure_model,
    )
    (patient_dir / "passport.md").write_text(
        f"# Patient {safe_patient_id}\n\nDe-identified source documents are in `documents.md`.\n",
        encoding="utf-8",
    )
    vault_path.write_text(
        json.dumps(document.vault, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _sync_memory(root)
    return patient_dir


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
    patient_dir = ingest_text(
        patient_id,
        extracted.text,
        root or config.runtime.patient_root,
        pii_detector=detector,
        structure_model=model,
    )
    return patient_dir, extracted.stats


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
    )


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


def _append_document(path: Path, text: str) -> None:
    clean = text.strip()
    if not clean:
        return
    if path.exists() and path.read_text(encoding="utf-8").strip():
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n---\n\n")
            handle.write(clean + "\n")
    else:
        path.write_text(clean + "\n", encoding="utf-8")


def _write_structured_files(patient_dir: Path, text: str, *, structure_model=None) -> None:
    structured = _prepass_structure(text)
    if structure_model is not None:
        structured = _merge_structure(structured, _model_structure(text, structured, structure_model))

    (patient_dir / "timeline").mkdir(exist_ok=True)
    (patient_dir / "labs").mkdir(exist_ok=True)
    _write_md(patient_dir / "timeline" / "events.md", "Timeline", structured["timeline"])
    _write_md(patient_dir / "problems.md", "Problems", structured["problems"])
    _write_md(patient_dir / "meds.md", "Medications", structured["meds"])
    _write_md(patient_dir / "labs" / "labs.md", "Labs", structured["labs"])
    _write_md(patient_dir / "strategy.md", "Strategy", ["Review changes with a human clinician."])


def _prepass_structure(text: str) -> dict[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "timeline": [line for line in lines if _DATE.search(line)],
        "problems": [_after_label(line, "Problem") for line in lines if line.lower().startswith("problem:")],
        "meds": [line for line in lines if _MED.search(line)],
        "labs": [line for line in lines if _LAB.search(line)],
    }


def _model_structure(text: str, prepass: dict[str, list[str]], structure_model) -> dict[str, list[str]]:
    system_prompt = (
        "Extract clinical structure from de-identified medical text. "
        "Return JSON object with array fields: timeline, problems, meds, labs. "
        "Do not infer PII and do not restore token values."
    )
    messages = [
        {
            "role": "user",
            "content": json.dumps({"text": text, "prepass": prepass}, ensure_ascii=False),
        }
    ]
    if hasattr(structure_model, "build_kwargs") and hasattr(structure_model, "call"):
        response = structure_model.call(structure_model.build_kwargs(messages, system_prompt, []))
        payload = getattr(response, "content", response)
    else:
        payload = structure_model(messages, system_prompt)
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("structuring model response must be a JSON object")
    return _coerce_structure(data)


def _coerce_structure(data: dict) -> dict[str, list[str]]:
    out = {}
    for key in ("timeline", "problems", "meds", "labs"):
        values = data.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"structuring model field must be a list: {key}")
        out[key] = [str(item).strip() for item in values if str(item).strip()]
    return out


def _merge_structure(first: dict[str, list[str]], second: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        key: _dedupe([*first.get(key, []), *second.get(key, [])])
        for key in ("timeline", "problems", "meds", "labs")
    }


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
