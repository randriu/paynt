"""paynt.utils.version_check: a warning for each way in which the Storm of payntbind differs from the Storm of stormpy."""

from types import SimpleNamespace

import pytest

import paynt.utils.version_check


def storm(origin_info, version="2.0.0", from_system=False, development=False, directory="/opt/storm"):
    return SimpleNamespace(
        storm_version=lambda: version,
        storm_from_system=lambda: from_system,
        storm_development_version=lambda: development,
        storm_origin_info=lambda: origin_info,
        storm_directory=lambda: directory,
    )


def payntbind_storm(commit="abc", **kwargs):
    return storm(("storm", commit), **kwargs)


def stormpy_storm(commit="abc", **kwargs):
    # stormpy names the branch too
    return storm(("storm", "master", commit), **kwargs)


def check(monkeypatch, caplog, payntbind, stormpy):
    """The warnings, and whether the Storms are said to be the same."""
    monkeypatch.setattr(paynt.utils.version_check, "payntbind", SimpleNamespace(info=payntbind))
    monkeypatch.setattr(paynt.utils.version_check, "stormpy", SimpleNamespace(info=stormpy))
    caplog.set_level("INFO")
    paynt.utils.version_check.check_stormpy_compatibility()
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    return warnings, "seem to be the same" in caplog.text


def test_the_same_storm_gives_no_warning(monkeypatch, caplog):
    assert check(monkeypatch, caplog, payntbind_storm(), stormpy_storm()) == ([], True)


@pytest.mark.parametrize(
    ("payntbind", "stormpy", "warning"),
    [
        ({"version": "1.9.0"}, {}, "Storm used by payntbind (1.9.0) and stormpy (2.0.0) are not the same!"),
        ({"from_system": True}, {}, "Storm used by payntbind (system) and stormpy (fetched) have different origins!"),
        ({}, {"from_system": True}, "Storm used by payntbind (fetched) and stormpy (system) have different origins!"),
        ({"development": True}, {}, "Storm used by payntbind (development) and stormpy (stable) have different development versions!"),
        ({}, {"development": True}, "Storm used by payntbind (stable) and stormpy (development) have different development versions!"),
        ({"commit": "def"}, {}, "Storm used by payntbind (('storm', 'def')) and stormpy (('storm', 'abc')) have different origin information!"),
        (
            {"from_system": True, "directory": "/usr/local"},
            {"from_system": True},
            "Storm used by payntbind is located at /usr/local while Storm used by stormpy is located at /opt/storm.",
        ),
    ],
)
def test_a_difference_gives_its_warning(monkeypatch, caplog, payntbind, stormpy, warning):
    assert check(monkeypatch, caplog, payntbind_storm(**payntbind), stormpy_storm(**stormpy)) == ([warning], False)


def test_a_storm_of_the_system_is_compared_by_its_directory_and_not_its_commit(monkeypatch, caplog):
    payntbind, stormpy = payntbind_storm(commit="def", from_system=True), stormpy_storm(from_system=True)
    assert check(monkeypatch, caplog, payntbind, stormpy) == ([], True)


def test_every_difference_is_reported(monkeypatch, caplog):
    warnings, same = check(monkeypatch, caplog, payntbind_storm(version="1.9.0", development=True), stormpy_storm())
    assert len(warnings) == 2 and not same
