import pytest


@pytest.fixture(autouse=True)
def _no_startup_recovery(monkeypatch):
    """App startup marks interrupted index jobs as failed in the real registry; never do that from tests."""
    from src.sources.jobs import index_jobs
    monkeypatch.setattr(index_jobs, "recover_interrupted", lambda: None)
