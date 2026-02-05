# LTX-2 CLI testing

## How to run tests

From the repository root:

```bash
# Unit tests only (no GPU, no network, no models)
uv run pytest packages/ltx-pipelines/tests/ -vv --tb=short --maxfail=1

# Stricter (warnings as errors)
uv run pytest packages/ltx-pipelines/tests/ -vv -Werror -W always --tb=short --maxfail=1

# Exclude integration tests (skip subprocess ltx --help etc.)
uv run pytest packages/ltx-pipelines/tests/ -m "not integration" -vv --tb=short --maxfail=1

# Include integration tests (requires uv and workspace install)
uv run pytest packages/ltx-pipelines/tests/ -vv --tb=short --maxfail=1
```

No GPU or downloaded models are required for unit tests or for integration tests that only run `ltx --help` and `ltx <subcommand> --help`.

## Test status matrix

| Area | Tests | Requires | Expected status |
|------|-------|----------|-----------------|
| model_resolve | resolve_model_path (local/HF mocks), resolve_args_paths | Nothing | pass |
| config | _flatten, _normalize_key, load_config (TOML/YAML), apply_config_to_parser | Nothing | pass |
| args | LoraAction use_raw_path, basic_arg_parser type, minimal parse | Nothing | pass |
| cli | Parser structure, two-phase parse, config overlay, help buffer | Nothing | pass |
| CLI help (integration) | ltx --help, ltx one-stage --help, ltx distilled --help | uv, workspace | pass |
| Full pipeline run | ltx one-stage ... with real paths | GPU, checkpoint, Gemma, output dir | manual / skip in CI |

## Unit test coverage

- **test_model_resolve.py**: Local path (file/dir) returns resolved path; `repo_id:filename` mocks `hf_hub_download`; repo-only mocks `snapshot_download`; `resolve_args_paths` leaves existing paths unchanged and resolves HF specs (checkpoint, lora).
- **test_config.py**: Key normalization and flatten; load_config from TOML/YAML; FileNotFoundError and bad extension; apply_config_to_parser sets defaults and CLI overrides.
- **test_args.py**: LoraAction raw vs resolved path; basic_arg_parser checkpoint type str vs resolve_path; default_1_stage and default_2_stage minimal parse.
- **test_cli.py**: Root parser has all subcommands; two-phase parse (subcommand + rest, subparser.parse_args(rest)); config file applied then CLI overrides; help output contains subcommands and --config.

## Integration tests

Tests marked `@pytest.mark.integration` run the installed `ltx` command via `uv run ltx ...`. They require the workspace to be installed (e.g. `uv sync` at repo root). Run them with the full suite or exclude with `-m "not integration"` for unit-only.

## Full pipeline run (manual)

To run a full generation (e.g. one-stage) you need:

- GPU with enough VRAM
- LTX-2 checkpoint (local or HuggingFace ID)
- Gemma text encoder (local or HF ID)
- Output path

Example (after `uv tool install .` or `uv run` from root):

```bash
ltx one-stage \
  --checkpoint-path Lightricks/LTX-2:ltx-2-19b-dev-fp8.safetensors \
  --gemma-root google/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A cat playing" \
  --output-path out.mp4
```

This is not run automatically in the test suite; document results manually if needed.
