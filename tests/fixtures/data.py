import pytest


@pytest.fixture
def minimal_session():
    """Path to a folder with the minimal session structure."""
    return "tests/data/minimal_session"
