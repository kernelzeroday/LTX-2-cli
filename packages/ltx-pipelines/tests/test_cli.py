from pathlib import Path
from unittest.mock import patch

import pytest

from ltx_pipelines.cli import _build_root_parser
from ltx_pipelines.utils.config import apply_config_to_parser, load_config


def test_build_root_parser_subcommands() -> None:
    root = _build_root_parser()
    subparsers_action = next(a for a in root._actions if getattr(a, "choices", None) is not None)
    choices = set(subparsers_action.choices)
    assert "one-stage" in choices
    assert "two-stages" in choices
    assert "distilled" in choices
    assert "ic-lora" in choices
    assert "keyframe-interp" in choices


def test_two_phase_parse_one_stage(tmp_path: Path) -> None:
    root = _build_root_parser()
    out = tmp_path / "out.mp4"
    out.touch()
    argv = [
        "one-stage",
        "--checkpoint-path", "c",
        "--gemma-root", "g",
        "--prompt", "p",
        "--output-path", str(out),
    ]
    args, rest = root.parse_known_args(argv)
    assert args.subcommand == "one-stage"
    assert args.prompt == "p"
    assert getattr(args, "checkpoint_path", None) == "c"
    assert getattr(args, "_run", None) == "one_stage"


def test_config_overlay_seed_from_config(tmp_path: Path) -> None:
    from ltx_pipelines.utils.config import _parser_defaults, apply_config_to_namespace
    root = _build_root_parser()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 99\nprompt: from_config\n")
    argv = [
        "--config", str(cfg),
        "one-stage",
        "--checkpoint-path", "c",
        "--gemma-root", "g",
        "--prompt", "p",
        "--output-path", str(tmp_path / "out.mp4"),
    ]
    args, rest = root.parse_known_args(argv)
    assert args.config == cfg
    subparsers_action = next(a for a in root._actions if getattr(a, "choices", None) is not None)
    sub_parser = subparsers_action.choices["one-stage"]
    defaults_before = _parser_defaults(sub_parser)
    config = load_config(args.config)
    apply_config_to_parser(sub_parser, config)
    full_args = sub_parser.parse_args(rest) if rest else args
    if not rest and config:
        apply_config_to_namespace(sub_parser, full_args, config, defaults_before)
    assert full_args.seed == 99
    assert full_args.prompt == "p"


def test_config_overlay_cli_overrides_config(tmp_path: Path) -> None:
    from ltx_pipelines.utils.config import _parser_defaults, apply_config_to_namespace
    root = _build_root_parser()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 99\n")
    argv = [
        "--config", str(cfg),
        "one-stage",
        "--checkpoint-path", "c",
        "--gemma-root", "g",
        "--prompt", "p",
        "--output-path", str(tmp_path / "out.mp4"),
        "--seed", "1",
    ]
    args, rest = root.parse_known_args(argv)
    subparsers_action = next(a for a in root._actions if getattr(a, "choices", None) is not None)
    sub_parser = subparsers_action.choices["one-stage"]
    defaults_before = _parser_defaults(sub_parser)
    config = load_config(args.config)
    apply_config_to_parser(sub_parser, config)
    full_args = sub_parser.parse_args(rest) if rest else args
    if not rest and config:
        apply_config_to_namespace(sub_parser, full_args, config, defaults_before)
    assert full_args.seed == 1


def test_cli_help_stdout() -> None:
    import io
    root = _build_root_parser()
    buf = io.StringIO()
    root.print_help(buf)
    out = buf.getvalue()
    assert "one-stage" in out
    assert "two-stages" in out
    assert "--config" in out


import subprocess
import sys
_has_triton = subprocess.run(
    [sys.executable, "-c", "import triton"],
    capture_output=True,
    timeout=5,
).returncode == 0


@pytest.mark.integration
@pytest.mark.skipif(not _has_triton, reason="triton not available (e.g. no GPU env)")
def test_ltx_help_subprocess() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        ["uv", "run", "ltx", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root_dir,
    )
    assert r.returncode == 0
    assert "one-stage" in r.stdout
    assert "two-stages" in r.stdout


@pytest.mark.integration
@pytest.mark.skipif(not _has_triton, reason="triton not available (e.g. no GPU env)")
def test_ltx_one_stage_help_subprocess() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        ["uv", "run", "ltx", "one-stage", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root_dir,
    )
    assert r.returncode == 0
    assert "--checkpoint-path" in r.stdout or "--prompt" in r.stdout


@pytest.mark.integration
@pytest.mark.skipif(not _has_triton, reason="triton not available (e.g. no GPU env)")
def test_ltx_distilled_help_subprocess() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        ["uv", "run", "ltx", "distilled", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root_dir,
    )
    assert r.returncode == 0
    assert "distilled" in r.stdout or "--prompt" in r.stdout
