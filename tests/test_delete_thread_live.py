# tests/test_delete_thread_live.py
from copy import deepcopy

def _cfg(cfg_base, thread_id: str, checkpoint_ns: str) -> dict:
    cfg = deepcopy(cfg_base)
    cfg["configurable"]["thread_id"] = thread_id
    cfg["configurable"]["checkpoint_ns"] = checkpoint_ns
    cfg["configurable"].pop("checkpoint_id", None)
    return cfg

def _put_ck(saver, cfg, ck_id: str, parent_id: str | None = None):
    # Minimal checkpoint dict your saver can serialize.
    # ts can be anything; you’re storing it as a string.
    ck = {
        "id": ck_id,
        "ts": "2025-12-12T00:00:00+00:00",
        "v": 1,
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": [],
    }
    md = {"source": "delete-test"}
    if parent_id is not None:
        cfg2 = deepcopy(cfg)
        cfg2["configurable"]["checkpoint_id"] = parent_id
        return saver.put(cfg2, ck, md, new_versions={})
    return saver.put(cfg, ck, md, new_versions={})

def test_delete_thread_removes_everything(saver, cfg_base):
    cfg = _cfg(cfg_base, thread_id="t-del-1", checkpoint_ns="demo-del")

    # create 2 checkpoints so timeline has >1
    cfg1 = _put_ck(saver, cfg, "ck1")
    cfg2 = _put_ck(saver, cfg1, "ck2", parent_id="ck1")

    # add writes for latest checkpoint
    saver.put_writes(cfg2, [("messages", {"x": 1})], task_id="task-1")

    # Preconditions: we can read latest and list returns something
    assert saver.get_tuple(cfg) is not None
    assert len(list(saver.list(cfg))) >= 1

    tid = cfg["configurable"]["thread_id"]
    cns = cfg["configurable"]["checkpoint_ns"]

    # Strong pre-check: keys exist
    assert saver._get(saver._key_cp(tid, cns, "ck1")) is not None
    assert saver._get(saver._key_cp(tid, cns, "ck2")) is not None
    assert saver._get(saver._key_latest(tid, cns)) is not None
    assert saver._get(saver._key_timeline(tid, cns)) is not None

    # Act
    saver.delete_thread(cfg)

    # Postconditions: latest/timeline gone => get_tuple(None) returns None
    assert saver.get_tuple(cfg) is None
    assert list(saver.list(cfg)) == []

    # Strong checks: records removed
    assert saver._get(saver._key_cp(tid, cns, "ck1")) is None
    assert saver._get(saver._key_cp(tid, cns, "ck2")) is None
    assert saver._get(saver._key_writes(tid, cns, "ck1")) is None
    assert saver._get(saver._key_writes(tid, cns, "ck2")) is None
    assert saver._get(saver._key_latest(tid, cns)) is None
    assert saver._get(saver._key_timeline(tid, cns)) is None


def test_delete_thread_works_if_timeline_missing(saver, cfg_base):
    cfg = _cfg(cfg_base, thread_id="t-del-2", checkpoint_ns="demo-del")

    _put_ck(saver, cfg, "ck1")

    tid = cfg["configurable"]["thread_id"]
    cns = cfg["configurable"]["checkpoint_ns"]

    # Simulate timeline being missing/corrupt: remove it manually
    saver._remove(saver._key_timeline(tid, cns))

    # Delete should still remove checkpoint via latest pointer
    saver.delete_thread(cfg)

    assert saver._get(saver._key_cp(tid, cns, "ck1")) is None
    assert saver._get(saver._key_latest(tid, cns)) is None
