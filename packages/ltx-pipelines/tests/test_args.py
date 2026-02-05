import argparse
from pathlib import Path

import pytest

from ltx_pipelines.utils.args import (
    LoraAction,
    basic_arg_parser,
    default_1_stage_arg_parser,
    default_2_stage_arg_parser,
    resolve_path,
)


def test_lora_action_use_raw_path_true() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        use_raw_path=True,
        nargs="+",
        default=[],
    )
    args = parser.parse_args(["--lora", "path/to/lora.safetensors", "0.8"])
    assert len(args.lora) == 1
    assert args.lora[0].path == "path/to/lora.safetensors"
    assert args.lora[0].strength == 0.8


def test_lora_action_use_raw_path_false_resolves(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        use_raw_path=False,
        nargs="+",
        default=[],
    )
    args = parser.parse_args(["--lora", str(tmp_path), "0.8"])
    assert len(args.lora) == 1
    assert args.lora[0].path == str(tmp_path.resolve().as_posix())
    assert args.lora[0].strength == 0.8


def test_lora_action_path_only_uses_default_strength() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        use_raw_path=True,
        nargs="+",
        default=[],
    )
    args = parser.parse_args(["--lora", "lora.safetensors"])
    assert len(args.lora) == 1
    assert args.lora[0].path == "lora.safetensors"
    assert args.lora[0].strength == 1.0


def test_basic_arg_parser_cli_raw_checkpoint_type_is_str() -> None:
    parser = basic_arg_parser(cli_model_paths_raw=True)
    action = next(a for a in parser._actions if a.dest == "checkpoint_path")
    assert action.type is str


def test_basic_arg_parser_default_checkpoint_type_is_resolve_path() -> None:
    parser = basic_arg_parser(cli_model_paths_raw=False)
    action = next(a for a in parser._actions if a.dest == "checkpoint_path")
    assert action.type is resolve_path


def test_default_1_stage_arg_parser_parse_minimal(tmp_path: Path) -> None:
    parser = default_1_stage_arg_parser(cli_model_paths_raw=True)
    out = tmp_path / "out.mp4"
    out.touch()
    args = parser.parse_args([
        "--checkpoint-path", str(tmp_path),
        "--gemma-root", str(tmp_path),
        "--prompt", "p",
        "--output-path", str(out),
    ])
    assert args.checkpoint_path == str(tmp_path)
    assert args.gemma_root == str(tmp_path)
    assert args.prompt == "p"
    assert args.output_path == str(out.resolve().as_posix())
    assert args.seed == 10


def test_default_2_stage_arg_parser_parse_minimal(tmp_path: Path) -> None:
    parser = default_2_stage_arg_parser(cli_model_paths_raw=True)
    out = tmp_path / "out.mp4"
    out.touch()
    args = parser.parse_args([
        "--checkpoint-path", str(tmp_path),
        "--gemma-root", str(tmp_path),
        "--distilled-lora", str(tmp_path) + "/lora.safetensors",
        "--spatial-upsampler-path", str(tmp_path) + "/up.safetensors",
        "--prompt", "p",
        "--output-path", str(out),
    ])
    assert args.checkpoint_path == str(tmp_path)
    assert args.distilled_lora is not None
    assert args.spatial_upsampler_path == str(tmp_path) + "/up.safetensors"
    assert args.prompt == "p"
