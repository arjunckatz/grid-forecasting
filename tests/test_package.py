import delhi_grid


def test_package_import_exposes_version() -> None:
    assert delhi_grid.__version__ == "0.1.0"
