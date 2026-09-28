from .metrics import DenovoMetrics, compute_denovo_metrics, format_metrics

__all__ = [
    "DenovoMetrics",
    "compute_denovo_metrics",
    "evaluate_generative",
    "evaluate_teacher_forced",
    "format_metrics",
]


def __getattr__(name: str):
    if name == "evaluate_generative":
        from .evaluate import evaluate_generative

        return evaluate_generative
    if name == "evaluate_teacher_forced":
        from .evaluate import evaluate_teacher_forced

        return evaluate_teacher_forced
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
