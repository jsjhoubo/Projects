"""Contract suite for the IncidentTracker storage primitive."""
import pytest
from aegislab.enterprise.incident_tracker import IncidentTracker, stable_digest

@pytest.fixture
def store():
    return IncidentTracker(max_records=20)


def test_empty(store):
    assert store.count() == 0
    assert store.stats()["keys"] == 0
    assert store.get("a", "x") is None


def test_create(store):
    a = store.put("a", "x", {"n": 1})
    assert a.revision == 1
    assert a.payload == {"n": 1}
    assert a.digest == stable_digest({"n": 1})
    assert store.get("a", "x") == a


def test_versions(store):
    store.put("a", "x", {"n": 1})
    store.put("a", "x", {"n": 2})
    assert store.get("a", "x", 1).payload["n"] == 1
    assert store.get("a", "x").payload["n"] == 2
    assert store.get("a", "x", 99) is None
    assert [r.revision for r in store.history("a", "x")] == [1,2]


def test_optimistic_write(store):
    store.put("a", "x", {"n": 1})
    with pytest.raises(ValueError):
        store.put("a", "x", {"n": 2}, expected_revision=0)
    assert store.put("a", "x", {"n": 2}, expected_revision=1).revision == 2


def test_tenant_isolation(store):
    store.put("a", "x", {"n": 1})
    store.put("b", "x", {"n": 7})
    assert store.get("b", "x").payload["n"] == 7
    assert store.count("a") == 1
    assert store.list_keys("a") == ["x"]
    assert len(store.export("b")) == 1


def test_compare(store):
    store.put("a", "x", {"a": 1, "gone": True})
    store.put("a", "x", {"a": 2, "new": True})
    diff = store.compare("a", "x", 1, 2)
    assert diff["changed"]["a"] == {"before":1,"after":2}
    assert diff["added"] == {"new": True}
    assert diff["removed"] == {"gone": True}


def test_compare_missing(store):
    with pytest.raises(KeyError):
        store.compare("a", "missing", 1, 2)


def test_input_validation(store):
    with pytest.raises(ValueError):
        store.put("", "k", {})
    with pytest.raises(ValueError):
        store.put("tenant", "", {})
    with pytest.raises(TypeError):
        store.put("tenant", "k", "not a dict")


def test_mutation_isolation(store):
    data = {"nested": {"x": [1]}}
    store.put("a", "x", data)
    data["nested"]["x"].append(2)
    assert store.get("a", "x").payload["nested"]["x"] == [1]


def test_history_limit(store):
    for i in range(5):
        store.put("a", "x", {"i": i})
    assert len(store.history("a", "x", 3)) == 3
    assert store.history("a", "x", 0) == []
    with pytest.raises(ValueError):
        store.history("a", "x", -1)


def test_list_keys(store):
    for key in ["x:2", "x:1", "y:1"]:
        store.put("a", key, {})
    assert store.list_keys("a", "x:") == ["x:1", "x:2"]


def test_import(store):
    assert store.import_records("a", [{"key":"a","v":1}, {"key":"b","v":2}]) == 2
    assert store.count("a") == 2
    with pytest.raises(ValueError):
        store.import_records("a", [{"not_key":1}])


def test_capacity():
    s = IncidentTracker(max_records=1)
    s.put("a", "x", {})
    with pytest.raises(OverflowError):
        s.put("a", "x", {})


def test_delete_tenant(store):
    store.put("a", "x", {})
    store.put("b", "x", {})
    assert store.delete_tenant("a") == 1
    assert store.get("a", "x") is None
    assert store.count("b") == 1
