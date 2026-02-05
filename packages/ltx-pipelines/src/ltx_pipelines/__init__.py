"""
LTX-2 Pipelines: High-level video generation pipelines and utilities.
This package provides ready-to-use pipelines for video generation:
- TI2VidOneStagePipeline: Text/image-to-video in a single stage
- TI2VidTwoStagesPipeline: Two-stage generation with upsampling
- DistilledPipeline: Fast distilled two-stage generation
- ICLoraPipeline: Image/video conditioning with distilled LoRA
- KeyframeInterpolationPipeline: Keyframe-based video interpolation
- ModelLedger: Central coordinator for loading and building models
For more detailed components and utilities, import from specific submodules
like `ltx_pipelines.utils.media_io` or `ltx_pipelines.utils.constants`.
"""

__all__ = [
    "DistilledPipeline",
    "ICLoraPipeline",
    "KeyframeInterpolationPipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesPipeline",
]


def __getattr__(name: str):  # noqa: N807
    if name == "DistilledPipeline":
        from ltx_pipelines.distilled import DistilledPipeline
        return DistilledPipeline
    if name == "ICLoraPipeline":
        from ltx_pipelines.ic_lora import ICLoraPipeline
        return ICLoraPipeline
    if name == "KeyframeInterpolationPipeline":
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        return KeyframeInterpolationPipeline
    if name == "TI2VidOneStagePipeline":
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        return TI2VidOneStagePipeline
    if name == "TI2VidTwoStagesPipeline":
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        return TI2VidTwoStagesPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
