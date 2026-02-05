import argparse
import logging
import sys
from pathlib import Path

from ltx_pipelines.utils.args import (
    VideoConditioningAction,
    default_1_stage_arg_parser,
    default_2_stage_arg_parser,
    default_2_stage_distilled_arg_parser,
)
from ltx_pipelines.utils.config import (
    _parser_defaults,
    apply_config_to_namespace,
    apply_config_to_parser,
    load_config,
)
from ltx_pipelines.utils.model_resolve import resolve_args_paths


def _subparser_one_stage(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "one-stage",
        parents=[default_1_stage_arg_parser(cli_model_paths_raw=True)],
        help="Single-stage text/image-to-video generation.",
        conflict_handler="resolve",
    )
    parser.set_defaults(_run="one_stage")
    return parser


def _subparser_two_stages(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "two-stages",
        parents=[default_2_stage_arg_parser(cli_model_paths_raw=True)],
        help="Two-stage generation with upsampling and distilled LoRA refinement.",
        conflict_handler="resolve",
    )
    parser.set_defaults(_run="two_stages")
    return parser


def _subparser_distilled(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "distilled",
        parents=[default_2_stage_distilled_arg_parser(cli_model_paths_raw=True)],
        help="Fast two-stage distilled pipeline (8 sigmas stage 1, 4 stage 2).",
        conflict_handler="resolve",
    )
    parser.set_defaults(_run="distilled")
    return parser


def _subparser_ic_lora(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "ic-lora",
        parents=[default_2_stage_distilled_arg_parser(cli_model_paths_raw=True)],
        help="Video-to-video / image-to-video with IC-LoRA control (depth, pose, canny, etc.).",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
    )
    parser.set_defaults(_run="ic_lora")
    return parser


def _subparser_keyframe_interp(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "keyframe-interp",
        parents=[default_2_stage_arg_parser(cli_model_paths_raw=True)],
        help="Interpolate between keyframe images.",
        conflict_handler="resolve",
    )
    parser.set_defaults(_run="keyframe_interp")
    return parser


def _build_root_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ltx",
        description="LTX-2 video generation CLI. Use subcommands to choose a pipeline.",
    )
    root.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config file (YAML or TOML). CLI arguments override config.",
    )
    root.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory. Defaults to HF_HOME / standard cache.",
    )
    subparsers = root.add_subparsers(dest="subcommand", required=True)
    _subparser_one_stage(subparsers)
    _subparser_two_stages(subparsers)
    _subparser_distilled(subparsers)
    _subparser_ic_lora(subparsers)
    _subparser_keyframe_interp(subparsers)
    return root


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    root = _build_root_parser()
    args, rest = root.parse_known_args()
    config_path = getattr(args, "config", None)
    cache_dir = getattr(args, "cache_dir", None)
    subcommand = args.subcommand
    subparsers_action = next(a for a in root._actions if getattr(a, "choices", None) is not None)
    sub_parser = subparsers_action.choices[subcommand]
    defaults_before = _parser_defaults(sub_parser)
    config = load_config(config_path) if config_path is not None else None
    if config is not None:
        apply_config_to_parser(sub_parser, config)
    full_args = sub_parser.parse_args(rest) if rest else args
    if not rest and config is not None:
        apply_config_to_namespace(sub_parser, full_args, config, defaults_before)
    full_args.config = config_path
    full_args.cache_dir = cache_dir
    resolve_args_paths(full_args, cache_dir=cache_dir)
    run_name = full_args._run
    if run_name == "one_stage":
        from ltx_pipelines.ti2vid_one_stage import _run_one_stage
        _run_one_stage(full_args)
    elif run_name == "two_stages":
        from ltx_pipelines.ti2vid_two_stages import _run_two_stages
        _run_two_stages(full_args)
    elif run_name == "distilled":
        from ltx_pipelines.distilled import _run_distilled
        _run_distilled(full_args)
    elif run_name == "ic_lora":
        from ltx_pipelines.ic_lora import _run_ic_lora
        _run_ic_lora(full_args)
    elif run_name == "keyframe_interp":
        from ltx_pipelines.keyframe_interpolation import _run_keyframe_interp
        _run_keyframe_interp(full_args)
    else:
        root.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
