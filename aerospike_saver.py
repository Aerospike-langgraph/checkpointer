# aerospike_saver.py
import json, time, aerospike
from typing import Any, Dict, Optional, Tuple
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

SEP = "|"

def _ids_from_config(config: dict) -> Tuple[str, str, Optional[str]]:
    c = (config or {}).get("configurable", {})
    if not c.get("thread_id"):
        raise ValueError("configurable.thread_id is required")
    return c["thread_id"], c.get("checkpoint_ns") or "default", c.get("checkpoint_id")

class AerospikeSaverMin(BaseCheckpointSaver):
    """Smallest viable saver: get_tuple + put. No listing, no writes."""
    def __init__(self, host="127.0.0.1", port=3000, namespace="test"):
        self.ns = namespace
        self.set_ckpt = "lg_checkpoints"
        self.set_latest = "lg_latest"
        self.client = aerospike.client({"hosts": [(host, port)]}).connect()

    def _ck_key(self, thread: str, ns: str, ck: str):  # user key
        return (self.ns, self.set_ckpt, SEP.join([thread, ns, ck]))

    def _latest_key(self, thread: str, ns: str):
        return (self.ns, self.set_latest, SEP.join(["latest", thread, ns]))

    # LangGraph will call this after each super-step
    def put(self, config: dict, checkpoint: Dict[str, Any], metadata: Dict[str, Any],
            task_id: str | None = None, **kwargs) -> None:
        thread, ns, ck = _ids_from_config(config)
        if ck is None:
            ck = str(int(time.time() * 1000))  # simple monotonic id

        now = int(time.time() * 1000)
        bins = {
            "thread_id": thread,
            "checkpoint_ns": ns,
            "checkpoint_id": ck,
            "state_blob": json.dumps(checkpoint),
            "metadata_json": json.dumps({**(metadata or {}), "task_id": task_id}),
            "created_at": now,
        }
        # create-only so retries don't overwrite
        self.client.put(self._ck_key(thread, ns, ck), bins,
                        policy={"exists": aerospike.POLICY_EXISTS_CREATE})
        # keep a tiny "latest" pointer for quick resume
        self.client.put(self._latest_key(thread, ns),
                        {"thread_id": thread, "checkpoint_ns": ns,
                         "checkpoint_id": ck, "updated_at": now})
    def put_writes(self, config, writes, metadata, **kwargs):
        """
        Minimal implementation so LangGraph can call us without crashing.
        Store nothing (Week-0), or uncomment the Aerospike writes if you want.
        """
        return 

    # LangGraph asks this BEFORE running to see if there’s prior state
    def get_tuple(self, config: dict, **kwargs) -> Optional[CheckpointTuple]:
        thread, ns, ck = _ids_from_config(config)
        if ck is None:
            # resolve "latest"
            try:
                _, _, b = self.client.get(self._latest_key(thread, ns))
                ck = b["checkpoint_id"]
            except aerospike.exception.RecordNotFound:
                return None

        try:
            _, _, b = self.client.get(self._ck_key(thread, ns, ck))
        except aerospike.exception.RecordNotFound:
            return None

        checkpoint = json.loads(b["state_blob"])
        metadata = json.loads(b.get("metadata_json", "{}"))
        if not isinstance(checkpoint, dict) or "v" not in checkpoint:
        # malformed/legacy record: treat as no prior state
            return None
        
        parent = None
        pending_writes = []
        task_id = None

        # Build positionally based on arity
        fields = getattr(CheckpointTuple, "_fields", None)
        arity = len(fields) if fields else 5

        if arity >= 5:
    # (checkpoint, metadata, parent_checkpoint_id, pending_writes, task_id)
            return CheckpointTuple(checkpoint, metadata, parent, pending_writes, task_id)
        elif arity == 4:
    # Some older builds: (checkpoint, metadata, parent_checkpoint_id, pending_writes)
            return CheckpointTuple(checkpoint, metadata, parent, pending_writes)
        else:
    # Very old (unlikely), fall back to minimal
            return CheckpointTuple(checkpoint, metadata)
