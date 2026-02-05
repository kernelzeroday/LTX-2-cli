import argparse
from pathlib import Path

import pytest

from ltx_pipelines.utils.config import (
    _flatten,
    _normalize_key,
    _parser_defaults,
    apply_config_to_namespace,
    apply_config_to_parser,
    load_config,
)


def test_normalize_key() -> None:
    assert _normalize_key("a-b") == "a_b"
    assert _normalize_key("seed") == "seed"


def test_flatten_flat_dict() -> None:
    d = {"prompt": "x", "seed": 42}
    out = _flatten(d)
    assert out == {"prompt": "x", "seed": 42}


def test_flatten_nested_one_level() -> None:
    d = {"video_guider_params": {"cfg_scale": 3.0, "stg_scale": 1.0}}
    out = _flatten(d)
    assert "video_guider_params_cfg_scale" in out
    assert "video_guider_params_stg_scale" in out
    assert out["video_guider_params_cfg_scale"] == 3.0
    assert out["video_guider_params_stg_scale"] == 1.0


def test_load_config_toml(tmp_path: Path, sample_config_toml: str) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(sample_config_toml)
    data = load_config(cfg)
    assert data["prompt"] == "A test prompt"
    assert data["seed"] == 42
    assert data["output_path"] == "/tmp/out.mp4"


def test_load_config_yaml(tmp_path: Path, sample_config_yaml: str) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(sample_config_yaml)
    data = load_config(cfg)
    assert data["prompt"] == "A test prompt"
    assert data["seed"] == 42
    assert data["output_path"] == "/tmp/out.mp4"


def test_load_config_yml_extension(tmp_path: Path, sample_config_yaml: str) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(sample_config_yaml)
    data = load_config(cfg)
    assert data["seed"] == 42


def test_load_config_nonexistent() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_bad_extension(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_config(cfg)


def test_load_config_empty_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    data = load_config(cfg)
    assert isinstance(data, dict)


def test_apply_config_to_parser_sets_default() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    apply_config_to_parser(parser, {"seed": 99})
    args = parser.parse_args([])
    assert args.seed == 99


def test_apply_config_to_parser_cli_overrides() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    apply_config_to_parser(parser, {"seed": 99})
    args = parser.parse_args(["--seed", "1"])
    assert args.seed == 1


def test_apply_config_to_parser_ignores_unknown_key() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    apply_config_to_parser(parser, {"seed": 99, "unknown_key": "x"})
    args = parser.parse_args([])
    assert args.seed == 99
    assert not hasattr(args, "unknown_key")


def test_parser_defaults() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    out = _parser_defaults(parser)
    assert out.get("seed") == 10
    assert out.get("prompt") is None


def test_apply_config_to_namespace_fills_default_only() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="p")
    defaults_before = _parser_defaults(parser)
    ns = argparse.Namespace(seed=10, prompt="cli_prompt")
    apply_config_to_namespace(parser, ns, {"seed": 99, "prompt": "from_config"}, defaults_before)
    assert ns.seed == 99
    assert ns.prompt == "cli_prompt"
