# Decisions

## 2026-07-09 - De-ID Boundary

- Regex de-ID remains for structured identifiers.
- Free-text PII must go through a local Ollama detector before patient memory or model prompts.
- `deidentification.ollama_model` is intentionally unset; the operator chooses it before real patient documents are allowed.
- Date of birth is stored in the local vault and rendered in de-identified text as age.
- Non-synthetic ingest is fail-closed: `runtime.allow_real_patient_docs=true`, a configured model and a reachable detector are required.

## 2026-07-09 - Architecture Scope

- Konsilium stays a pipeline library now.
- Do not add the full agent loop, skills, tool dispatcher or control HTTP+SSE during Stage 1/2.
- Agent loop and scheduled autonomous monitoring move to Stage 3.
- Stage 2 still needs real hybrid memory over canonical patient Markdown before continuing model reviews on larger histories.

## 2026-07-09 - Patient Memory

- Canonical patient state stays as Markdown under `patients/<patient_id>/`.
- `PatientMemory` indexes document metadata and deterministic local vectors.
- Runtime dependency is embedded LanceDB; JSON metadata is only a local fallback when LanceDB is unavailable in a test environment.
- Retrieval is always scoped by `patient_id` before model prompts are assembled.

## 2026-07-09 - Ingest Structuring

- Structured patient files are extracted from already de-identified text.
- Non-synthetic ingest must use the configured model path for structuring.
- Regex lab/med matching remains only as a cheap pre-pass and synthetic-test fallback, not as the real-document extractor.

## 2026-07-09 - PDF Extraction

- PDF ingest first uses local text-layer extraction, then local OCR for weak pages via `ocrmypdf`/Tesseract (`deu+eng`) in the Docker image.
- Extraction fails loudly if any page remains empty after OCR; stats are PII-free and returned to operator surfaces.
- Photo/image ingest can reuse the same Tesseract path later; PDF is the only default OCR input for this stage.

## 2026-07-10 - De-ID Residue Gate

- Every ingest scans de-identified text for DOB, German street/PLZ, phone, long digit, KVNR and case-number residue before any patient-memory or vault write.
- Each named residue pattern has a configurable `block`, `report` or `ignore` action; defaults block ingest and errors reveal only line numbers and pattern names.
- `deid-preview` writes local preview, local vault and PII-free residue report under `previews/`; it does not structure, index or ingest the document.
- The local Ollama detector uses overlapping text chunks and deduplicates entities across chunks; defaults are 900 characters with 150 character overlap.

## 2026-07-09 - Public Snapshot Hygiene

- Public releases are sanitized working-tree snapshots, not private history pushes.
- The public leak gate is case-insensitive and blocks internal ecosystem identifiers.
- The first public snapshot was force-republished as a clean root commit after the gate was tightened.
- `scripts/publish.sh` uses `Sanitized snapshot: <private commit subject>` by default;
  `KONSILIUM_PUBLIC_MESSAGE` can name a release explicitly.
