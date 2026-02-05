import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ltx_core.loader import LoraPathStrengthAndSDOps


def resolve_model_path(spec: str, cache_dir: str | None = None) -> str:
    expanded = Path(spec).expanduser()
    if expanded.exists():
        return str(expanded.resolve().as_posix())
    if ":" in spec:
        repo_id, filename = spec.split(":", 1)
        local_path = hf_hub_download(
            repo_id=repo_id.strip(),
            filename=filename.strip(),
            cache_dir=cache_dir,
            local_files_only=False,
        )
        return local_path
    local_dir = snapshot_download(
        repo_id=spec.strip(),
        cache_dir=cache_dir,
        local_files_only=False,
    )
    return local_dir


def _resolve_path_attr(args: argparse.Namespace, attr: str, cache_dir: str | None) -> None:
    val = getattr(args, attr, None)
    if val is None:
        return
    if Path(val).expanduser().exists():
        return
    setattr(args, attr, resolve_model_path(val, cache_dir=cache_dir))


def resolve_args_paths(args: argparse.Namespace, cache_dir: str | None = None) -> None:
    _resolve_path_attr(args, "checkpoint_path", cache_dir)
    _resolve_path_attr(args, "gemma_root", cache_dir)
    _resolve_path_attr(args, "spatial_upsampler_path", cache_dir)
    for list_attr in ("lora", "distilled_lora"):
        loras = getattr(args, list_attr, None)
        if not loras:
            continue
        resolved = []
        for item in loras:
            path = item.path
            if not Path(path).expanduser().exists():
                path = resolve_model_path(path, cache_dir=cache_dir)
            resolved.append(LoraPathStrengthAndSDOps(path, item.strength, item.sd_ops))
        setattr(args, list_attr, resolved)
