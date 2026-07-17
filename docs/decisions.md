# Decisions

## 2026-07-09 - De-ID Boundary

- Regex de-ID remains for structured identifiers.
- Free-text PII passes through local GLiNER NER plus an Ollama second opinion before patient memory or model prompts; detected entities are unioned.
- Detector models are intentionally unset in the generic config; the operator chooses at least one local detector before real patient documents are allowed.
- Date of birth is stored in the local vault and rendered in de-identified text as age.
- Non-synthetic ingest is fail-closed: `runtime.allow_real_patient_docs=true`, a configured model and a reachable detector are required.

## 2026-07-09 - Architecture Scope

- Konsilium stays a pipeline library now.
- The shipped boundary is a pipeline library with CLI and MCP surfaces, not a
  general autonomous agent loop or control HTTP+SSE service.
- Scheduled autonomous monitoring remains future work; patient-scoped hybrid
  memory and multi-role reviews are implemented.

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
- Local GLiNER and Ollama detectors use overlapping text chunks and union deduplicated entities; defaults are 900 characters with 150 character overlap.

## 2026-07-10 - Address Policy

- Institutional addresses and phone/fax contacts remain plain text when nearby letterhead markers identify a public institution; private address context has priority and ambiguous addresses are tokenized.
- Physician names remain PERSON entities even on institutional letterheads.
- The detector ignores generic German/English role words and one-letter PERSON values; only proper-name-like PERSON values are accepted.
- Rationale: institutional contact details are public clinical context, while private identity is not.

## 2026-07-10 - Token-Safe Replacement

- Detector entities are applied longest-first as whole values and never inside existing `[KIND_n]` spans.
- Entity values shorter than three word characters are ignored at the de-ID boundary, independent of detector behavior.
- Residue blocks malformed or unbalanced token brackets, referral/page-header surnames, spelled German DOBs and OCR-split numeric years.

## 2026-07-10 - German-Only Scope

- Konsilium accepts German healthcare documents and produces German doctor-letter drafts only.
- The de-ID rules are based on German document conventions (`geb.`, `wh.`, `Pat.-Nr.` and letterhead formulas), and German Arztbrief style is the only real-material workflow that is currently testable.
- English and Russian medical-document support is intentionally not advertised or implemented because it has not been verified on real material.

## 2026-07-10 - OCR DOB and Contact Verification

- Any `Geburtsdatum` or `geb.` line without a nearby `age <n>` replacement blocks ingest, independent of OCR date format.
- Institutional email retention is an explicit de-ID decision passed to the residue gate; the gate blocks every other remaining email and does not run its own institutional heuristic.
- Compact physician names remain PERSON data; public institutional footer identifiers such as Ust-ID remain plain context.

## 2026-07-10 - Reviewed Preview Ingest

- DOB lines containing a patient token followed by an unpunctuated bare word block as partial-name residue.
- DOB marker words and an existing `age N` are excluded from partial-name residue detection.
- Deterministic `age N` replacements are protected spans; the subsequent detector pass cannot replace any part of them.
- Regex- and detector-sourced DOB values use the same age conversion; the observed `JuËI` OCR month is normalized to July.
- Six-digit case numbers following a DOB age in page footers are tokenized deterministically before detector output is applied.
- Only generic institutional mailboxes may remain plain; person-named mailboxes are tokenized even on institutional domains.
- OCR-spaced postal codes follow the same private/institutional context policy as ordinary postal codes.
- `ingest --from-preview` accepts only a locally edited
  `previews/preview-*.md` and reloads its preview vault.
- `--accept-residue` may downgrade explicitly named patterns to report-only for
  that single reviewed ingest; the acceptance is persisted to the document
  boundary and is never global configuration.

## 2026-07-17 - Scoped Tokens, Templates, and Review Inbox

- Numbered patient IDs scope identity tokens across documents; non-numeric IDs
  retain the documented unscoped fallback.
- The JSON identity vault remains machine-readable and a local Markdown token
  ledger gives the operator one review surface.
- Each stored source is rendered through the clinical document template with
  metadata and sections; accepted residue names are recorded in frontmatter.
- `preview-inbox` is an idempotent review-first pass over newly dropped files.

## 2026-07-17 - DIN 5008 Letter Channels

- Tokenized paper and e-mail drafts use separate DIN 5008 layouts.
- Real identity substitution is deterministic, local, returned to stdout, and
  never written back into patient memory.

## 2026-07-10 - Blocking Model Deadlines

- The stale timeout measures silence between streaming chunks only.
- Blocking JSON and subscription-provider calls use `request_timeout_s` as their sole watchdog deadline.

## 2026-07-10 - Per-Document Patient Memory

- Each ingest writes one canonical Markdown file under `patients/<id>/documents/`; the ingest APIs return that path.
- The existing structuring call also extracts document date, institutional topic, and institutional sender for the filename.
- Unsafe metadata falls back to neutral ASCII slugs; missing document dates use the ingest date and record `date_source: ingest` in frontmatter.
- Existing aggregate clinical files merge new structured entries, while the memory index retrieves the individual source documents.

## 2026-07-10 - Recipient Address Precedence

- Postal addresses in recipient blocks are private even when an institutional letterhead marker is nearby.
- A patient token adjacent to an address prevents institutional retention unless the block is clearly an institutional affiliation.
- The deterministic tokenizer and residue-gate exemption use the same recipient classification.
- Model-sourced ADDRESS values are validated before global substitution; clinical words and institutional city mentions are rejected unchanged.
- Digits alone are not address evidence; ADDR requires street, house-number, or PLZ structure, while age/unit/kinship spans are hard rejects.
- Rejected model entities are recorded in preview reports only as PII-free kind/reason pairs.
- Model-sourced PATIENT values remain privacy-first, but strong institutional, quantified-medical, and technical evidence prevents clinical or letterhead text from being replaced as a person.
- Model-sourced EMAIL values must contain `@`; PHONE values must have a numeric phone shape.
- Model-sourced DOB values require a birth marker at each replacement span; letter and event dates remain dates.
- Multi-line ADDRESS entities are validated and substituted one line at a time, so greeting text cannot be absorbed into an address token.
- Deterministic PLZ/city patterns are line-bound and reject any remaining cross-line capture before substitution.
- Compound `place/place, DD.MM.YYYY` letterhead lines are retained consistently while the same toponym remains tokenizable in patient-residence context.

## 2026-07-09 - Public Snapshot Hygiene

- Public releases are sanitized working-tree snapshots, not private history pushes.
- The public leak gate is case-insensitive and blocks internal ecosystem identifiers.
- The first public snapshot was force-republished as a clean root commit after the gate was tightened.
- `scripts/publish.sh` uses `Sanitized snapshot: <private commit subject>` by default;
  `KONSILIUM_PUBLIC_MESSAGE` can name a release explicitly.
