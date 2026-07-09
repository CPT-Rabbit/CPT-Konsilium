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
    ingest.add_argument("--file", required=True)
    ingest.add_argument("--synthetic", action="store_true", help="explicit synthetic/test ingest")

    review = subparsers.add_parser("review", help="run a consilium review")
    review.add_argument("--patient", required=True)
    review.add_argument("--roles", help="comma-separated roles")
    review.add_argument("--question")

    letter = subparsers.add_parser("letter", help="write a tokenized doctor-letter draft")
    letter.add_argument("--patient", required=True)
    letter.add_argument("--language", required=True, choices=["de", "en", "ru"])

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
    if args.command == "review":
        _review(config, args)
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
    from .ingest import extract_text_with_stats, ingest_patient_file, ingest_text

    if args.synthetic:
        extracted = extract_text_with_stats(args.file)
        patient_dir = ingest_text(
            args.patient,
            extracted.text,
            config.runtime.patient_root,
            allow_synthetic=True,
        )
        stats = extracted.stats
    else:
        patient_dir, stats = ingest_patient_file(
            config,
            args.patient,
            args.file,
            config.runtime.patient_root,
        )
    print(json.dumps({"patient_dir": str(patient_dir), "extraction": stats}, ensure_ascii=False))


def _review(config: Config, args: argparse.Namespace) -> None:
    from .review import case_review

    report = case_review(
        config.runtime.patient_root,
        args.patient,
        roles=_csv(args.roles) or None,
        question=args.question,
        model_client=_model_client(config),
    )
    report_path = config.runtime.patient_root / "patients" / args.patient / "consilium" / "latest.json"
    print(f"report={report_path}")
    print(report.get("chair_summary", ""))


def _letter(config: Config, args: argparse.Namespace) -> None:
    from .letters import doctor_letter

    print(doctor_letter(config.runtime.patient_root, args.patient, language=args.language))


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
