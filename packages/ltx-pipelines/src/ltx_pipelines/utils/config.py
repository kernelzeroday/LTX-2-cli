import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import yaml


def _normalize_key(k: str) -> str:
    return str(k).replace("-", "_")


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, val in d.items():
        nk = _normalize_key(key)
        full = f"{prefix}{nk}" if prefix else nk
        if isinstance(val, dict) and not any(isinstance(v, (dict, list)) for v in val.values()):
            for k2, v2 in val.items():
                out[f"{full}_{_normalize_key(k2)}"] = v2
        else:
            out[full] = val
    return out


def load_config(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf == ".toml":
        with path.open("rb") as f:
            data = tomllib.load(f)
    elif suf in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text()) or {}
    else:
        raise ValueError(f"Unsupported config extension: {suf}. Use .toml, .yaml, or .yml")
    return _flatten(dict(data))


def apply_config_to_parser(parser: argparse.ArgumentParser, config: dict) -> None:
    dests = {a.dest for a in parser._actions if hasattr(a, "dest") and a.dest != argparse.SUPPRESS}
    defaults = {k: v for k, v in config.items() if k in dests}
    if defaults:
        parser.set_defaults(**defaults)


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
    return {a.dest: a.default for a in parser._actions if getattr(a, "dest", None) is not None}


def apply_config_to_namespace(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    config: dict,
    defaults_before: dict,
) -> None:
    for k, v in config.items():
        if k not in defaults_before:
            continue
        if getattr(namespace, k, None) == defaults_before.get(k):
            setattr(namespace, k, v)
