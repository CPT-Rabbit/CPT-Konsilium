from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="konsilium")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--ping", action="store_true", help="one model connectivity check")
    parser.add_argument("--codex-login", action="store_true", help="authorize ChatGPT subscription")
    parser.add_argument("--stage1-smoke", action="store_true", help="synthetic ingest/de-ID smoke")
    parser.add_argument("--knowledge-smoke", help="offline knowledge-tool URL/header smoke query")
    parser.add_argument("--mcp", action="store_true", help="run MCP stdio server")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="ingest a patient document")
    ingest.add_argument("--patient", required=True)
    ingest_source = ingest.add_mutually_exclusive_group(required=True)
    ingest_source.add_argument("--file")
    ingest_source.add_argument("--from-preview")
    ingest.add_argument("--synthetic", action="store_true", help="explicit synthetic/test ingest")
    ingest.add_argument(
        "--accept-residue",
        help="comma-separated residue patterns the operator reviewed and accepted (from-preview only)",
    )

    preview = subparsers.add_parser("deid-preview", help="write a local de-identification preview without ingesting")
    preview.add_argument("--file", required=True)
    preview.add_argument(
        "--known-identity",
        help="patient id or identity-vault path whose confirmed PII is swept deterministically",
    )

    inbox = subparsers.add_parser("preview-inbox", help="de-identify every new inbox document into a review preview")
    inbox.add_argument("--patient", required=True, help="patient id whose vault seeds the known-identity sweep")
    inbox.add_argument("--dir", help="inbox directory (default: <patient_root>/inbox)")

    review = subparsers.add_parser("review", help="run a consilium review")
    review.add_argument("--patient", required=True)
    review.add_argument("--roles", help="comma-separated roles")
    review.add_argument("--question")

    letter = subparsers.add_parser("letter", help="write a tokenized doctor-letter draft")
    letter.add_argument("--patient", required=True)
    letter.add_argument("--channel", choices=("paper", "email"), default="paper")

    render = subparsers.add_parser("letter-render", help="render PII to stdout only")
    render.add_argument("--patient", required=True)
    render.add_argument("--file", required=True)

    monitor = subparsers.add_parser("monitor", help="write a monitor report")
    monitor.add_argument("--patients", required=True, help="comma-separated patient ids")

    memory = subparsers.add_parser("memory-search", help="search patient memory")
    memory.add_argument("--patient", required=True)
    memory.add_argument("--query", required=True)
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    os.environ.setdefault(config.auth.file_env, str(config.auth.default_path))

    if args.codex_login:
        _codex_login()
        return
    if args.ping:
        _ping(config)
        return
    if args.stage1_smoke:
        _stage1_smoke(config)
        return
    if args.knowledge_smoke:
        _knowledge_smoke(args.knowledge_smoke)
        return
    if args.mcp:
        from .mcp_server import run_stdio

        run_stdio(config)
        return
    if args.command == "ingest":
        _ingest(config, args)
        return
    if args.command == "deid-preview":
        _deid_preview(config, args)
        return
    if args.command == "review":
        _review(config, args)
        return
    if args.command == "preview-inbox":
        _preview_inbox(config, args)
        return
    if args.command == "letter":
        _letter(config, args)
        return
    if args.command == "letter-render":
        _letter_render(config, args)
        return
    if args.command == "monitor":
        _monitor(config, args)
        return
    if args.command == "memory-search":
        _memory_search(config, args)
        return
    parser.error("specify a command or legacy smoke flag")


def _ping(config: Config) -> None:
    model = _model_client(config)
    kwargs = model.build_kwargs(
        [{"role": "user", "content": "Reply with the single word: OK"}],
        "You are a connectivity health check.",
        [],
    )
    response = model.call(kwargs)
    print(f"PROVIDER={config.model.provider} MODEL={config.model.model}")
    print(f"REPLY: {response.content[:300]!r}")


def _codex_login() -> None:
    from .providers.codex import codex_device_login
    from .providers.credential_pool import AuthStore

    def notify(challenge: dict[str, str]) -> None:
        print(f"Open: {challenge['verification_url']}", flush=True)
        print(f"Code: {challenge['user_code']}", flush=True)
        print("Waiting for browser authorization...", flush=True)

    codex_device_login(AuthStore(), notify=notify)
    print("Codex device authorization saved.", flush=True)


def _stage1_smoke(config: Config) -> None:
    import json

    from .smoke import stage1_smoke

    print(json.dumps(stage1_smoke(config.runtime.patient_root), ensure_ascii=False, indent=2))


def _knowledge_smoke(query: str) -> None:
    import json

    from .smoke import knowledge_smoke

    print(json.dumps(knowledge_smoke(query), ensure_ascii=False, indent=2))


def _ingest(config: Config, args: argparse.Namespace) -> None:
    from .ingest import extract_text_with_stats, ingest_from_preview, ingest_patient_file, ingest_text

    if args.from_preview:
        if args.synthetic:
            raise ValueError("--synthetic cannot be combined with --from-preview")
        document_path = ingest_from_preview(
            config,
            args.patient,
            args.from_preview,
            config.runtime.patient_root,
            accepted_residue=_csv(getattr(args, "accept_residue", None)),
        )
        stats = {"kind": "reviewed_preview"}
    elif getattr(args, "accept_residue", None):
        raise ValueError("--accept-residue applies to reviewed previews only (--from-preview)")
    elif args.synthetic:
        extracted = extract_text_with_stats(args.file)
        document_path = ingest_text(
            args.patient,
            extracted.text,
            config.runtime.patient_root,
            allow_synthetic=True,
        )
        stats = extracted.stats
    else:
        document_path, stats = ingest_patient_file(
            config,
            args.patient,
            args.file,
            config.runtime.patient_root,
        )
    print(json.dumps({
        "patient_dir": str(document_path.parents[1]),
        "document_path": str(document_path),
        "extraction": stats,
    }, ensure_ascii=False))


def _deid_preview(config: Config, args: argparse.Namespace) -> None:
    from pathlib import Path

    from .ingest import _read_vault, _safe_id, deid_preview

    known_identity = None
    ref = getattr(args, "known_identity", None)
    if ref:
        path = Path(ref)
        if not path.exists():
            path = Path(config.runtime.patient_root) / "identity_vault" / f"{_safe_id(ref)}.json"
        known_identity = _read_vault(path) or None
    print(json.dumps(deid_preview(config, args.file, known_identity=known_identity), ensure_ascii=False))


def _preview_inbox(config: Config, args: argparse.Namespace) -> None:
    from pathlib import Path

    from .ingest import _read_vault, _safe_id, deid_preview, inbox_documents_to_preview

    root = Path(config.runtime.patient_root)
    inbox_dir = Path(args.dir) if args.dir else root / "inbox"
    known_identity = _read_vault(root / "identity_vault" / f"{_safe_id(args.patient)}.json") or None
    todo = inbox_documents_to_preview(inbox_dir, root / "previews")
    results = []
    for path in todo:
        report = deid_preview(config, path, known_identity=known_identity)["residue"]
        results.append({"file": path.name, "blocked": report["blocked"]})
    print(json.dumps({
        "inbox": str(inbox_dir),
        "previewed": len(results),
        "blocked": [item["file"] for item in results if item["blocked"]],
        "clean": [item["file"] for item in results if not item["blocked"]],
    }, ensure_ascii=False))


def _review(config: Config, args: argparse.Namespace) -> None:
    from .review import case_review

    report = case_review(
        config.runtime.patient_root,
        args.patient,
        roles=_csv(args.roles) or None,
        question=args.question,
        model_client=_model_client(config),
        residue_policy=config.deidentification.residue,
    )
    report_path = config.runtime.patient_root / "patients" / args.patient / "consilium" / "latest.json"
    print(f"report={report_path}")
    print(report.get("chair_summary", ""))


def _letter(config: Config, args: argparse.Namespace) -> None:
    from .letters import doctor_letter

    print(doctor_letter(config.runtime.patient_root, args.patient, channel=args.channel))


def _letter_render(config: Config, args: argparse.Namespace) -> None:
    from .letters import render_doctor_letter

    draft = Path(args.file).read_text(encoding="utf-8")
    print(render_doctor_letter(config.runtime.patient_root, args.patient, draft), end="")


def _monitor(config: Config, args: argparse.Namespace) -> None:
    from .monitor import monitor_review

    report = monitor_review(config.runtime.patient_root, _csv(args.patients))
    path = config.runtime.patient_root / "monitor" / "latest.json"
    print(f"report={path}")
    print(json.dumps({"patients": [item["patient_id"] for item in report["patients"]]}, ensure_ascii=False))


def _memory_search(config: Config, args: argparse.Namespace) -> None:
    from .memory import PatientMemory

    memory = PatientMemory(config.runtime.patient_root)
    hits = memory.search(args.query, patient_id=args.patient)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


def _model_client(config: Config):
    from .model_client import ModelClient
    from .providers.base import build_provider

    return ModelClient(
        build_provider(config.model),
        request_timeout_s=config.model.request_timeout_s,
        stream=config.model.stream,
    )


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
