import pytest


@pytest.fixture(autouse=True)
def _no_seed(monkeypatch):
    monkeypatch.setenv("PASZEL_NO_SEED", "1")  # UI tests use empty or synthetic states unless a test says otherwise
