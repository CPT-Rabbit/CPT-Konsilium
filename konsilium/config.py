from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ProviderKind = Literal["custom", "xai-oauth", "codex", "claude-cli"]


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderKind = "custom"
    model: str = "workers-ai/@cf/google/gemma-4-26b-a4b-it"
    base_url: str | None = None
    api_key_env: str = "CF_AIG_TOKEN"
    request_timeout_s: float = 120.0
    stream: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    root: Path = Path("/memory")
    collection: str = "konsilium_memory"


@dataclass(frozen=True)
class RuntimeConfig:
    patient_root: Path = Path("/memory")
    allow_real_patient_docs: bool = False


@dataclass(frozen=True)
class DeidentificationConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    timeout_s: float = 300.0


@dataclass(frozen=True)
class AuthConfig:
    file_env: str = "KONSILIUM_AUTH_FILE"
    default_path: Path = Path("/auth/auth.json")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    deidentification: DeidentificationConfig = field(default_factory=DeidentificationConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        try:
            import yaml
        except ModuleNotFoundError:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls.model_validate(data)

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def model_validate(cls, data: dict | None) -> "Config":
        data = data or {}
        return cls(
            model=ModelConfig(**data.get("model", {})),
            memory=MemoryConfig(**_paths(data.get("memory", {}), "root")),
            runtime=RuntimeConfig(**_paths(data.get("runtime", {}), "patient_root")),
            deidentification=DeidentificationConfig(**data.get("deidentification", {})),
            auth=AuthConfig(**_paths(data.get("auth", {}), "default_path")),
        )


def _paths(data: dict, *keys: str) -> dict:
    out = dict(data)
    for key in keys:
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    return out
