"""Controlled incentive, activity, and field-capacity benchmark extension."""


def run_commercial_review_pipeline(*args, **kwargs):
    """Lazily import the full orchestration path for a clean module CLI."""
    from .pipeline import run_commercial_review_pipeline as run

    return run(*args, **kwargs)


__all__ = ["run_commercial_review_pipeline"]
