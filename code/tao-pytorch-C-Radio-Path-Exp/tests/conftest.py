import pytest

def pytest_unconfigure(config):
    if hasattr(config, "testmon_data"):
        config.testmon_data.db.con.close()