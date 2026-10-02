import pytest
from fastapi import Request

from src.auth.store import User

TEST_ADMIN = User(id=1, username="test-admin", role="admin", must_change_password=False)


@pytest.fixture(autouse=True)
def _no_startup_side_effects(monkeypatch):
    """App startup must never touch the real data/ folder from tests."""
    from src.sources.jobs import index_jobs
    from src.auth import store as auth_store
    monkeypatch.setattr(index_jobs, "recover_interrupted", lambda: None)
    if not hasattr(auth_store.UserStore, "real_ensure_initial_admins"):
        auth_store.UserStore.real_ensure_initial_admins = auth_store.UserStore.ensure_initial_admins
    monkeypatch.setattr(auth_store.UserStore, "ensure_initial_admins", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _signed_in_admin(request):
    """API tests run as a signed-in admin unless marked @pytest.mark.real_auth."""
    if request.node.get_closest_marker("real_auth"):
        yield
        return
    from src.api.main import app
    from src.auth import deps

    def as_admin(req: Request):
        req.state.user = TEST_ADMIN
        return TEST_ADMIN
    app.dependency_overrides[deps.require_user] = as_admin
    app.dependency_overrides[deps.require_user_admin_to_change] = as_admin
    yield
    app.dependency_overrides.clear()


def pytest_configure(config):
    config.addinivalue_line("markers", "real_auth: run without the signed-in admin override")
