import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.model_resolve import resolve_args_paths, resolve_model_path


def test_resolve_model_path_existing_dir(tmp_path: Path) -> None:
    out = resolve_model_path(str(tmp_path))
    assert out == str(tmp_path.resolve().as_posix())


def test_resolve_model_path_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "foo.safetensors"
    f.write_text("x")
    out = resolve_model_path(str(f))
    assert out == str(f.resolve().as_posix())


def test_resolve_model_path_repo_colon_filename() -> None:
    with patch("ltx_pipelines.utils.model_resolve.hf_hub_download") as m:
        m.return_value = "/cache/org/repo/file.safetensors"
        out = resolve_model_path("org/repo:file.safetensors")
        assert out == "/cache/org/repo/file.safetensors"
        m.assert_called_once()
        call_kw = m.call_args[1]
        assert call_kw["repo_id"] == "org/repo"
        assert call_kw["filename"] == "file.safetensors"


def test_resolve_model_path_repo_colon_filename_stripped() -> None:
    with patch("ltx_pipelines.utils.model_resolve.hf_hub_download") as m:
        m.return_value = "/cache/path"
        resolve_model_path("  org/repo  :  file.safetensors  ")
        call_kw = m.call_args[1]
        assert call_kw["repo_id"] == "org/repo"
        assert call_kw["filename"] == "file.safetensors"


def test_resolve_model_path_repo_colon_filename_cache_dir() -> None:
    with patch("ltx_pipelines.utils.model_resolve.hf_hub_download") as m:
        m.return_value = "/custom/cache/path"
        resolve_model_path("a/b:c.safetensors", cache_dir="/custom/cache")
        assert m.call_args[1]["cache_dir"] == "/custom/cache"


def test_resolve_model_path_repo_only() -> None:
    with patch("ltx_pipelines.utils.model_resolve.snapshot_download") as m:
        m.return_value = "/cache/org/repo"
        out = resolve_model_path("org/repo")
        assert out == "/cache/org/repo"
        m.assert_called_once()
        assert m.call_args[1]["repo_id"] == "org/repo"


def test_resolve_model_path_repo_only_cache_dir() -> None:
    with patch("ltx_pipelines.utils.model_resolve.snapshot_download") as m:
        m.return_value = "/cache"
        resolve_model_path("org/repo", cache_dir="/hf")
        assert m.call_args[1]["cache_dir"] == "/hf"


def test_resolve_args_paths_checkpoint_existing_unchanged(tmp_path: Path) -> None:
    ns = argparse.Namespace(checkpoint_path=str(tmp_path), gemma_root=str(tmp_path))
    with patch("ltx_pipelines.utils.model_resolve.resolve_model_path") as m:
        resolve_args_paths(ns)
        m.assert_not_called()
    assert ns.checkpoint_path == str(tmp_path)


def test_resolve_args_paths_checkpoint_hf_resolved(tmp_path: Path) -> None:
    ns = argparse.Namespace(
        checkpoint_path="org/repo:ckpt.safetensors",
        gemma_root=str(tmp_path),
        spatial_upsampler_path=None,
        lora=[],
        distilled_lora=None,
    )
    with patch("ltx_pipelines.utils.model_resolve.resolve_model_path") as m:
        m.return_value = "/resolved/ckpt.safetensors"
        resolve_args_paths(ns)
        m.assert_called_with("org/repo:ckpt.safetensors", cache_dir=None)
    assert ns.checkpoint_path == "/resolved/ckpt.safetensors"


def test_resolve_args_paths_lora_resolved(tmp_path: Path) -> None:
    lora_item = LoraPathStrengthAndSDOps("org/repo:lora.safetensors", 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
    ns = argparse.Namespace(
        checkpoint_path=str(tmp_path),
        gemma_root=str(tmp_path),
        spatial_upsampler_path=None,
        lora=[lora_item],
        distilled_lora=None,
    )
    with patch("ltx_pipelines.utils.model_resolve.resolve_model_path") as m:
        m.return_value = "/resolved/lora.safetensors"
        resolve_args_paths(ns)
    assert len(ns.lora) == 1
    assert ns.lora[0].path == "/resolved/lora.safetensors"
    assert ns.lora[0].strength == 0.8
    assert ns.lora[0].sd_ops is LTXV_LORA_COMFY_RENAMING_MAP
