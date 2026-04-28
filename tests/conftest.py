from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only.

    The default AnyIO parametrization also exercises Trio, but this CI/container
    environment does not permit Trio's socket buffer setup.
    """

    return "asyncio"
