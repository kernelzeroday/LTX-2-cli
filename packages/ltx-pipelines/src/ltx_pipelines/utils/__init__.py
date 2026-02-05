__all__ = ["ModelLedger"]


def __getattr__(name: str):  # noqa: N807
    if name == "ModelLedger":
        from ltx_pipelines.utils.model_ledger import ModelLedger
        return ModelLedger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
