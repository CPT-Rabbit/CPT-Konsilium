# Deploy - Konsilium

One image supports both targets. Host differences live in config, env files and
bind mounts.

## Mac (Docker Desktop)

Default local layout, adjust paths as needed:

```text
~/konsilium/
  config/config.yaml          # copy from config.mac.yaml.example, no secrets
  secrets/konsilium.env       # chmod 600, CF_AIG_TOKEN and optional gateway key
  auth/auth.json              # Codex subscription device-login store
  memory/                     # patients/, identity_vault/, lance/, monitor/
```

Prepare folders and config:

```sh
export KONSILIUM_HOME="$HOME/konsilium"
mkdir -p "$KONSILIUM_HOME"/{config,memory,secrets,auth}
cp config.mac.yaml.example "$KONSILIUM_HOME/config/config.yaml"
chmod 700 "$KONSILIUM_HOME/secrets" "$KONSILIUM_HOME/auth"
touch "$KONSILIUM_HOME/secrets/konsilium.env"
chmod 600 "$KONSILIUM_HOME/secrets/konsilium.env"
```

Safely add the CF gateway token for `model.provider: custom`:

```sh
set -eu
export KONSILIUM_HOME="${KONSILIUM_HOME:-$HOME/konsilium}"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
grep -v '^CF_AIG_TOKEN=' "$KONSILIUM_HOME/secrets/konsilium.env" > "$tmp" || true
printf 'CF_AIG_TOKEN: ' >&2
stty -echo
IFS= read -r TOK
stty echo
printf '\n' >&2
printf 'CF_AIG_TOKEN=%s\n' "$TOK" >> "$tmp"
unset TOK
install -m 600 "$tmp" "$KONSILIUM_HOME/secrets/konsilium.env"
```

Build and run smokes:

```sh
export KONSILIUM_HOME="$HOME/konsilium"
docker compose -f docker-compose.mac.yml build
docker compose -f docker-compose.mac.yml run --rm konsilium --config /config/config.yaml --ping
docker compose -f docker-compose.mac.yml run --rm konsilium --config /config/config.yaml --stage1-smoke
docker compose -f docker-compose.mac.yml run --rm konsilium --config /config/config.yaml --knowledge-smoke "metformin hba1c"
```

Host inspection after `--stage1-smoke`:

```sh
find "$KONSILIUM_HOME/memory/patients" -maxdepth 3 -type f -print
find "$KONSILIUM_HOME/memory/patients/synthetic-stage1/documents" -name '*.md' -maxdepth 1 -print
ls "$KONSILIUM_HOME/memory/identity_vault"
```

PDF ingest handles text-layer and scanned pages locally. The image includes
`ocrmypdf` plus Tesseract `deu` and `eng`; no cloud OCR is used. If OCR is
needed but unavailable, ingest fails before writing an empty document.

Run a preview before admitting a new real document source:

```sh
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml deid-preview --file /memory/inbox/befund.pdf
find "$KONSILIUM_HOME/memory/previews" -maxdepth 1 -type f -print
```

This writes a tokenized preview, a local preview vault and a PII-free residue
report under `memory/previews/`. It never creates patient memory, runs
structuring or calls a reasoning provider. Residue hits are reported only as
pattern names and line numbers; real ingest remains blocked until they are
resolved.

For OCR residue that cannot be repaired automatically, edit the generated
`preview-*.md` locally, remove every flagged value, then ingest that reviewed
file:

```sh
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml ingest --patient case-1 \
  --from-preview /memory/previews/preview-befund.md
```

The command accepts only files under `memory/previews/`, loads the neighboring
`.vault.json`, and runs the residue gate again before structuring or memory
writes. There is no flag that bypasses the gate.

Synthetic operator flow:

```sh
cat > "$KONSILIUM_HOME/memory/synthetic-letter.txt" <<'EOF'
Patient: Anna Mueller
2026-07-01 HbA1c 8.1, started Metformin 500 mg.
Problem: Type 2 diabetes
EOF

docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml ingest --patient cli-smoke --file /memory/synthetic-letter.txt --synthetic
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml review --patient cli-smoke --roles internist,endocrinologist
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml letter --patient cli-smoke
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml letter-render --patient cli-smoke \
  --file /memory/patients/cli-smoke/letters/doctor_letter_de.md
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml memory-search --patient cli-smoke --query "HbA1c diabetes"
```

`letter-render` prints PII to stdout only. Do not redirect it into `/memory`.

### MCP stdio bridge

Claude Desktop / Claude Code use the MCP server, not a custom UI. The MCP tool
surface intentionally omits `letter-render`; it cannot read `identity_vault/`.
Use `deid_preview` before `ingest_document` for a new real document source;
the tool returns paths and a PII-free residue report, never vault content.

Create a host wrapper:

```sh
mkdir -p "$KONSILIUM_HOME/bin"
cat > "$KONSILIUM_HOME/bin/konsilium-mcp" <<'EOF'
#!/bin/sh
set -eu
: "${KONSILIUM_HOME:=$HOME/konsilium}"
exec docker run -i --rm \
  --user 10000:10000 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --env-file "$KONSILIUM_HOME/secrets/konsilium.env" \
  -e KONSILIUM_AUTH_FILE=/auth/auth.json \
  -e HOME=/home/konsilium \
  -v "$KONSILIUM_HOME/config:/config:ro" \
  -v "$KONSILIUM_HOME/memory:/memory" \
  -v "$KONSILIUM_HOME/secrets:/secrets:ro" \
  -v "$KONSILIUM_HOME/auth:/auth" \
  -v "$HOME/.claude:/home/konsilium/.claude:ro" \
  konsilium:latest --config /config/config.yaml --mcp
EOF
chmod 700 "$KONSILIUM_HOME/bin/konsilium-mcp"
```

If Claude Code auth lives in `~/.claude.json`, add that mount to the wrapper.

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "konsilium": {
      "command": "/home/operator/konsilium/bin/konsilium-mcp"
    }
  }
}
```

Claude Code `.mcp.json`:

```json
{
  "mcpServers": {
    "konsilium": {
      "command": "/home/operator/konsilium/bin/konsilium-mcp"
    }
  }
}
```

Expected chat flow: call `deid_preview`, call `ingest_document` with `synthetic=true`, run
`case_review` with two roles, inspect context through `memory_search` /
`memory_get`, then call `doctor_letter` for a tokenized draft. Rendered PII
stays CLI-only on the host.

### Mac provider switches

Provider is a config-only choice in `model.provider`; review/ingest paths do not
change.

`custom` uses the Cloudflare AI Gateway and `CF_AIG_TOKEN` from
`secrets/konsilium.env`.

`codex` uses the inherited subscription provider. Run device login once with the
bind-mounted auth store, then switch config to `model.provider: codex`:

```sh
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml --codex-login
docker compose -f docker-compose.mac.yml run --rm konsilium \
  --config /config/config.yaml --ping
```

`claude-cli` is Mac-local only. The image installs Claude Code CLI and
`docker-compose.mac.yml` bind-mounts host credentials from:

```text
~/.claude -> /home/konsilium/.claude:ro
```

If local Claude Code stores auth in `~/.claude.json`, add the commented
mount in `docker-compose.mac.yml` before running. Then set:

```yaml
model:
  provider: claude-cli
  model: "sonnet"
```

and verify:

```sh
docker compose -f docker-compose.mac.yml run --rm konsilium --config /config/config.yaml --ping
```

No MCP inference bridge is used. MCP remains for tools, not for sourcing model
completions from native apps.

Ollama for de-identification runs natively on the Mac host. Container config must
use:

```yaml
deidentification:
  ollama_url: "http://host.docker.internal:11434"
  ollama_model: "gemma3:4b"
  timeout_s: 300.0
```

The generic server example keeps `ollama_model` unset; the Mac example uses
`gemma3:4b` for the local PII detector.

Docker Desktop has no app-server `DOCKER-USER` firewall equivalent. Mac-side
controls are the code egress guard, tokens-only memory, and provider config
pointing at the gateway or local subscription backends. This is not an air-gap.

## app-server

Supported Stage 3+ hardened variant. Same image, server config and mounts:

```text
/opt/konsilium/
  config/config.yaml
  secrets/konsilium.env
  auth/auth.json
  memory/
  runtime/
```

Server defaults stay gateway-based:

```yaml
model:
  provider: custom
  base_url: "https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/compat"
  api_key_env: CF_AIG_TOKEN
```

Run with the hardened container pattern where the server firewall/proxy exists:

```sh
docker run -d --name konsilium --restart unless-stopped \
  --user 10000:10000 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --memory=2g --cpus=1 \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --env-file /opt/konsilium/secrets/konsilium.env \
  -e KONSILIUM_AUTH_FILE=/auth/auth.json \
  -v /opt/konsilium/config:/config:ro \
  -v /opt/konsilium/secrets:/secrets:ro \
  -v /opt/konsilium/auth:/auth \
  -v /opt/konsilium/memory:/memory \
  ghcr.io/cpt-rabbit/konsilium:latest \
  --config /config/config.yaml --help
```

Attach the server egress-allowlist outside the container: LLM gateway, PubMed,
Semantic Scholar and approved guideline hosts. Subscription providers
(`codex`, `claude-cli`) are Mac-local operator choices; app-server keeps gateway
providers unless the operator explicitly changes that decision.

Safe server paste for the gateway key:

```sh
set -eu
sudo install -d -m 700 /opt/konsilium/secrets
umask 077
tmp="$(mktemp)"
oldstty=""
trap '[ -n "${oldstty:-}" ] && stty "$oldstty" || true; rm -f "$tmp"' EXIT
if sudo test -f /opt/konsilium/secrets/konsilium.env; then
  sudo grep -v '^KONSILIUM_GATEWAY_AGENT_KEY=' /opt/konsilium/secrets/konsilium.env > "$tmp" || true
fi
oldstty="$(stty -g)"
printf 'KONSILIUM_GATEWAY_AGENT_KEY: ' >&2
stty -echo
IFS= read -r KEY
stty "$oldstty"
printf '\n' >&2
printf 'KONSILIUM_GATEWAY_AGENT_KEY=%s\n' "$KEY" >> "$tmp"
unset KEY
sudo install -m 600 "$tmp" /opt/konsilium/secrets/konsilium.env
sudo chown root:root /opt/konsilium/secrets/konsilium.env
sudo ls -l /opt/konsilium/secrets/konsilium.env
```
