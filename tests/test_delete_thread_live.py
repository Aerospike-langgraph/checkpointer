# tests/test_delete_thread_live.py
from copy import deepcopy

def _cfg(cfg_base, thread_id: str, checkpoint_ns: str) -> dict:
    cfg = deepcopy(cfg_base)
    cfg["configurable"]["thread_id"] = thread_id
    cfg["configurable"]["checkpoint_ns"] = checkpoint_ns
    cfg["configurable"].pop("checkpoint_id", None)
    return cfg

def _put_ck(saver, cfg, ck_id: str, parent_id: str | None = None):
    ck = {
        "id": ck_id,
        "ts": "2025-12-12T00:00:00+00:00",
        "v": 1,
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": [],
    }
    md = {"source": "sync-delete-test"}

    if parent_id is not None:
        cfg2 = deepcopy(cfg)
        cfg2["configurable"]["checkpoint_id"] = parent_id
        return saver.put(cfg2, ck, md, new_versions={})

    return saver.put(cfg, ck, md, new_versions={})


def test_delete_thread_removes_everything(saver, cfg_base):
    cfg = _cfg(cfg_base, thread_id="t-del-sync-1", checkpoint_ns="demo-del-sync")

    cfg1 = _put_ck(saver, cfg, "ck1")
    cfg2 = _put_ck(saver, cfg1, "ck2", parent_id="ck1")

    saver.put_writes(cfg2, [("messages", {"x": 1})], task_id="task-1")

    # Preconditions
    assert saver.get_tuple(cfg) is not None
    assert len(list(saver.list(cfg))) >= 1

    tid = cfg["configurable"]["thread_id"]
    cns = cfg["configurable"]["checkpoint_ns"]

    # Strong pre-checks: meta exists
    assert saver._get(saver._key_latest(tid, cns)) is not None
    assert saver._get(saver._key_timeline(tid, cns)) is not None

    # Act (use the name you implemented)
    if hasattr(saver, "delete_thread"):
        saver.delete_thread(cfg)
    else:
        saver.delete_history(cfg)

    # Postconditions
    assert saver.get_tuple(cfg) is None
    assert list(saver.list(cfg)) == []

    # Strong checks: records removed
    assert saver._get(saver._key_cp(tid, cns, "ck1")) is None
    assert saver._get(saver._key_cp(tid, cns, "ck2")) is None
    assert saver._get(saver._key_writes(tid, cns, "ck1")) is None
    assert saver._get(saver._key_writes(tid, cns, "ck2")) is None
    assert saver._get(saver._key_latest(tid, cns)) is None
    assert saver._get(saver._key_timeline(tid, cns)) is None


def test_delete_thread_ok_if_called_twice(saver, cfg_base):
    cfg = _cfg(cfg_base, thread_id="t-del-sync-2", checkpoint_ns="demo-del-sync")
    _put_ck(saver, cfg, "ck1")

    # First delete
    if hasattr(saver, "delete_thread"):
        saver.delete_thread(cfg)
        saver.delete_thread(cfg)  # should not raise
    else:
        saver.delete_history(cfg)
        saver.delete_history(cfg)

    assert saver.get_tuple(cfg) is None


def test_delete_thread_handles_missing_timeline(saver, cfg_base):
    cfg = _cfg(cfg_base, thread_id="t-del-sync-3", checkpoint_ns="demo-del-sync")
    _put_ck(saver, cfg, "ck1")

    tid = cfg["configurable"]["thread_id"]
    cns = cfg["configurable"]["checkpoint_ns"]

    # Simulate timeline missing
    saver._delete(saver._key_timeline(tid, cns))

    # Delete should still not error; (whether ck1 is deleted depends on whether
    # your delete implementation also checks __latest__)
    if hasattr(saver, "delete_thread"):
        saver.delete_thread(cfg)
    else:
        saver.delete_history(cfg)

    # At minimum, latest should be gone after delete
    assert saver._get(saver._key_latest(tid, cns)) is None
