"""Lab configuration loaded from lab.toml."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import dacite

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

LAB_CONFIG_FILENAME = "lab.toml"

_DACITE_CONFIG = dacite.Config(strict=True)


@dataclass(frozen=True)
class HostServiceConfig:
    """Optional host-side service started before the container."""

    command: str
    port: int


@dataclass(frozen=True)
class BackendConfig:
    """Backend configuration from lab.toml [backend]."""

    module: str = "backend.py"
    cls: str = "EvalBackend"
    host_service: HostServiceConfig | None = None


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox configuration from lab.toml [sandbox]."""

    dockerfile: str | None = None


@dataclass(frozen=True)
class LabConfig:
    """Parsed lab.toml configuration."""

    name: str = ""
    pipeline_dir: str = "pipeline"
    backend: BackendConfig = field(default_factory=BackendConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent_instructions: str = "AGENT.md"
    results_file: str = "results.tsv"

    @property
    def safe_name(self) -> str:
        """Name sanitized for use in Docker image tags, file paths, etc."""
        return re.sub(r"[^a-z0-9._-]+", "-", self.name.lower()).strip("-")

    @staticmethod
    def find_lab_root(start: Path | None = None) -> Path:
        """Walk up from start (default cwd) to find lab.toml."""
        current = (start or Path.cwd()).resolve()
        while True:
            candidate = current / LAB_CONFIG_FILENAME
            if candidate.exists():
                return current
            parent = current.parent
            if parent == current:
                raise FileNotFoundError(
                    f"No {LAB_CONFIG_FILENAME} found in {start or Path.cwd()} "
                    f"or any parent directory. Run 'arl init' to create one."
                )
            current = parent

    @staticmethod
    def load(lab_root: Path) -> LabConfig:
        """Load and parse lab.toml from a lab root directory."""
        config_path = lab_root / LAB_CONFIG_FILENAME
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        lab = data.get("lab", {})
        backend_data = data.get("backend", {})
        sandbox_data = data.get("sandbox", {})

        # Rename TOML key "class" to dataclass field "cls"
        if "class" in backend_data:
            backend_data["cls"] = backend_data.pop("class")

        # Default name to the directory name
        if "name" not in lab:
            lab["name"] = lab_root.name

        lab["backend"] = dacite.from_dict(
            BackendConfig, backend_data, config=_DACITE_CONFIG
        )
        lab["sandbox"] = dacite.from_dict(
            SandboxConfig, sandbox_data, config=_DACITE_CONFIG
        )

        return dacite.from_dict(LabConfig, lab, config=_DACITE_CONFIG)
