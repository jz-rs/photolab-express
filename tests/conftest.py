import pytest

from src import cache


@pytest.fixture(autouse=True)
def no_disk_cache():
    """Los tests prueban la lógica real, no lo que haya quedado en caché."""
    cache.set_enabled(False)
    yield
    cache.set_enabled(True)
