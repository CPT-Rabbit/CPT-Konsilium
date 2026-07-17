# Konsilium

**Privacy-first medical document analysis with multi-perspective AI review.**

Konsilium ingests medical documents (discharge letters, lab reports, clinical
findings), de-identifies them locally, maintains a longitudinal per-patient
memory, and runs multi-specialist "consilium" reviews that produce structured
reports: claims with evidence references, open questions for physicians,
explicit disagreements between specialist perspectives, and draft letters to
doctors. It is a preparation tool for talking to real physicians — not a
replacement for them.

> **Medical disclaimer.** Konsilium does not diagnose, prescribe, or provide
> medical advice. Every output is preparation material intended to be reviewed
> with a licensed physician. Do not make treatment decisions based on its
> reports.

## Why

If you (or someone you care for) manage a complex medical history, you
accumulate PDFs from different practices, labs, and hospitals. Getting value
out of them with an LLM normally means uploading raw documents — names,
addresses, insurance numbers and all — to a cloud provider. Konsilium is built
around a different contract:

**Personal identity never leaves your machine. Only de-identified medical
content ever reaches a model provider.**

## How privacy works

Two data planes, enforced in code — not by prompt instructions:

```
PDF / text document
      │
      ▼
┌───────────────────────────────┐
│ Local de-identification       │  regex (structured identifiers)
│                               │  + GLiNER NER (primary recall)
│                               │  + Ollama (second opinion)
└───────┬───────────────┬───────┘
        │               │
        ▼               ▼
  patients/<id>/    identity_vault/<id>.json
  de-identified     token → PII mapping
  Markdown memory   LOCAL ONLY, never indexed,
  [PATIENT_1]-style never sent to any API
  tokens
        │
        ▼
  LLM reasoning, literature search (PubMed, Semantic Scholar),
  consilium reviews — de-identified content only
```

- **Fail-closed ingest.** Non-synthetic ingest is impossible without a
  configured, reachable local PII detector and a structuring model. A
  misconfigured pipeline fails loudly instead of degrading to regex-only.
- **Egress guard.** Every outbound knowledge query is checked; anything
  containing PII tokens or raw-PII patterns is rejected before the request
  is built.
- **Tokens-only on disk.** Patient memory and letter drafts contain only
  tokens. Rendering a letter with real names happens in a local command that
  prints to stdout and never writes rendered PII into the memory tree.
- **Dates of birth become ages** in de-identified text; the original stays in
  the local vault.
- **Local embeddings.** Memory retrieval uses a deterministic local embedding —
  no embedding API calls.

## Features

- **Scope: German healthcare documents.** De-identification and document
  conventions are verified against German medical material only.
- **Ingest pipeline**: PDF/text → local ensemble de-ID → templated Markdown
  documents plus timeline, problem list, medications, and labs per patient.
- **Review-first inbox**: `preview-inbox` de-identifies newly dropped files
  idempotently before an operator admits a reviewed preview.
- **Scoped identity tokens**: tokens are unique per numbered patient; a
  local-only operator ledger lists token mappings beside the JSON vault.
- **Patient-scoped hybrid memory**: embedded LanceDB index over canonical
  Markdown files (plain-JSON fallback), retrieval always filtered by patient.
  The memory is human-readable — open it in Obsidian or any editor to see
  exactly what the system knows.
- **Consilium reviews**: each selected specialist role (Markdown profiles in
  `roles/` — internist, endocrinologist, neurologist, add your own) gets an
  independent model pass; a chair-synthesis pass merges them and surfaces
  real disagreements instead of smoothing them over.
- **DIN 5008 doctor letters in German**: tokenized paper and e-mail drafts on
  disk, with deterministic local-only PII rendering.
- **Literature grounding**: PubMed (NCBI E-utilities) and Semantic Scholar
  search with the egress guard in front; AWMF guideline lookup.
- **Monitoring**: periodic multi-patient review reports.
- **MCP server**: drive everything from Claude Desktop or any MCP client —
  chat is the UI, no custom frontend to run.
- **Pluggable model providers**: any OpenAI-compatible endpoint (e.g.
  Cloudflare AI Gateway), ChatGPT subscription (device login), or a local
  Claude Code CLI in headless mode.

## Quickstart (Docker)

```sh
git clone <this-repo> && cd konsilium
cp config.yaml.example config.yaml   # edit: model provider, paths
docker build -t konsilium .

# smoke checks
docker run --rm -v "$PWD/config:/config:ro" konsilium \
  --config /config/config.yaml --stage1-smoke
docker run --rm -v "$PWD/config:/config:ro" konsilium \
  --config /config/config.yaml --knowledge-smoke "metformin hba1c"
```

See `DEPLOY.md` for the full local (Docker Desktop on macOS, bind-mounted
memory folders, Ollama on the host) and hardened server runbooks.

### De-identification model

Install the local detector dependencies, pull an
[Ollama](https://ollama.com) model, and configure both ensemble layers:

```yaml
deidentification:
  gliner_model: "urchade/gliner_multi_pii-v1"
  ollama_url: "http://host.docker.internal:11434"
  ollama_model: "qwen3:4b"
```

Ingest of real documents is deliberately blocked until a de-ID model is
configured **and** `runtime.allow_real_patient_docs` is set to `true`. Review
the de-identified output on synthetic and first real documents before trusting
the pipeline — you can read every file it writes.

## Usage

### CLI

```sh
konsilium() { docker run --rm -i \
  -v "$HOME/konsilium/config:/config:ro" \
  -v "$HOME/konsilium/memory:/memory" \
  --env-file "$HOME/konsilium/secrets/konsilium.env" \
  konsilium --config /config/config.yaml "$@"; }

konsilium preview-inbox --patient case-1
konsilium deid-preview --file /memory/inbox/befund.pdf --known-identity case-1
konsilium ingest --patient case-1 --from-preview /memory/previews/preview-befund.md
# After reviewing named non-PII residue patterns for this preview only:
konsilium ingest --patient case-1 --from-preview /memory/previews/preview-befund.md \
  --accept-residue DIGIT_RUN
konsilium review --patient case-1 --roles internist,endocrinologist \
  --question "What should the next appointment clarify?"
konsilium letter --patient case-1 --channel paper
konsilium letter-render --patient case-1 \
  --file patients/case-1/letters/doctor_letter_de_paper.md   # PII to stdout only
konsilium memory-search --patient case-1 --query "HbA1c trend"
konsilium monitor --patients case-1,case-2
```

### MCP (chat as the interface)

Register the server in Claude Desktop (`claude_desktop_config.json`) or any
MCP client — snippet in `DEPLOY.md`. Exposed tools: `ingest_document`, `deid_preview`,
`case_review`, `doctor_letter`, `memory_search`, `memory_get`,
`monitor_review`, `list_patients`.

Then just talk: *"Ingest the lab report from the inbox for case-1 and run an
internist + endocrinologist review."*

By design, the MCP surface has **no** tool that returns vault contents or
rendered PII — letter rendering stays a local CLI command.

### Model providers

| Provider | Config `model.provider` | Use case |
|---|---|---|
| OpenAI-compatible endpoint | `custom` | Cloudflare AI Gateway, any compatible API |
| ChatGPT subscription | `codex` | local use via device login (`--codex-login`) |
| Claude Code CLI (headless) | `claude-cli` | local use via your Claude subscription |

The de-identification boundary is identical for all providers: they only ever
see de-identified content.

## Memory layout

```
memory/
  patients/<id>/
    passport.md          # summary
    documents/           # one de-identified Markdown file per source document
    timeline/events.md   # dated events
    problems.md  meds.md  labs/labs.md
    hypotheses/  consilium/  letters/   # reports & tokenized drafts
    strategy.md
  identity_vault/<id>.json   # local-only token→PII map
  identity_vault/<id>.tokens.md  # local-only operator token ledger
  inbox/  previews/          # review-first source queue and local artifacts
  lance/                     # embedded vector index
```

Everything canonical is plain Markdown. Point Obsidian at `memory/` for a
zero-code dashboard of what the system has stored.
Ingest returns the stored document path. Source files use
`YYYY-MM-DD_Topic_Sender.md`; frontmatter records when the date had to fall
back to the ingest date.

## Development

```sh
pip install -e ".[dev]"
python -m pytest          # suite must pass with and without lancedb installed
```

Design decisions and known implementation limits are recorded in
`docs/decisions.md` and neutral engineering comments.

## Status & roadmap

Early, actively developed. Working today: local ensemble de-ID, PDF OCR,
review-first inbox previews, templated patient memory, consilium reviews,
DIN 5008 letter channels, PubMed/Semantic Scholar/AWMF knowledge tools, CLI,
MCP server, and Docker deployment. Planned: scheduled autonomous monitoring
and research-agent delegation for deep literature work.

## License

Apache-2.0
