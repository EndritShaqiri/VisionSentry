from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root from {current}. Expected a parent directory containing 'src' and 'configs'."
    )


def resolve_project_path(
    value: str | Path,
    project_root: Path,
    *,
    must_exist: bool = False,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path

