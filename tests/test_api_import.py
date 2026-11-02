import paynt


def test_get_version_returns_a_non_empty_string():
    assert isinstance(paynt.get_version(), str)
    assert paynt.get_version() != ""
