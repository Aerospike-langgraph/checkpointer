from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aerospike
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    Checkpoint,
    CheckpointMetadata,
    ChannelVersions,
)

SEP = "|"


def _now_ns() -> int:
    return time.time_ns()


class AerospikeSaver(BaseCheckpointSaver):
    """
    Minimal checkpointer with zero server-side querying requirements.

    Public API expects a RunnableConfig-like dict:

        config = {
            "configurable": {
                "thread_id": "<required>",
                "checkpoint_ns": "<required>",
                # optional:
                "checkpoint_id": "<for put/get or explicit resume>",
                "before": "<for list()>",
            },
            # you can also pass tags/metadata/etc. but they're ignored here
        }

    Storage layout (all in a single namespace):
      - main records (set=self.set_cp):
            key:  "{thread_id}|{checkpoint_ns}|{checkpoint_id}"
            bins: {
                "thread_id": str,
                "checkpoint_ns": str,
                "checkpoint_id": str,
                "checkpoint": str (JSON),
                "metadata":   str (JSON),
                "ts":         int (ns since epoch),
            }

      - latest pointer record (set=self.set_meta):
            key:  "{thread_id}|{checkpoint_ns}|__latest__"
            bins: { "checkpoint_id": str, "ts": int }

      - timeline record (set=self.set_meta):
            key:  "{thread_id}|{checkpoint_ns}|__timeline__"
            bins: { "items": str(JSON list[[ts:int, checkpoint_id:str], ...]) }

      - writes (optional, set=self.set_writes):
            key:  "{thread_id}|{checkpoint_ns}|{checkpoint_id}"
            bins: { "writes": str(JSON) }
    """

    def __init__(
        self,
        client: aerospike.Client,
        namespace: str = "test",
        set_cp: str = "lg_cp",
        set_writes: str = "lg_cp_w",
        set_meta: str = "lg_cp_meta",
        ttl: Optional[int] = None,
        timeline_max: int = 500,
    ) -> None:
        self.client = client
        self.ns = namespace
        self.set_cp = set_cp
        self.set_writes = set_writes
        self.set_meta = set_meta
        self.ttl = ttl
        self.timeline_max = max(1, int(timeline_max))

    # ---------- config parsing ----------
    @staticmethod
    def _ids_from_config(config: Optional[Dict[str, Any]]) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Returns (thread_id, checkpoint_ns, checkpoint_id, before)
        """
        cfg = (config or {})
        c = cfg.get("configurable", {}) or {}
        md = cfg.get("metadata", {}) or {}

        # 1) thread_id: must exist somewhere
        thread_id = c.get("thread_id") or md.get("thread_id")
        if not thread_id:
            raise ValueError("configurable.thread_id is required in RunnableConfig")

        # 2) checkpoint_ns: optional, default ""
        checkpoint_ns = (
            c.get("checkpoint_ns")
            or md.get("checkpoint_ns")
            or ""
        )

        checkpoint_id = c.get("checkpoint_id")
        before = c.get("before")

        return thread_id, checkpoint_ns, checkpoint_id, before

    # ---------- keys ----------
    def _key_cp(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_cp, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_writes, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_latest(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__latest__")

    def _key_timeline(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__timeline__")

    # ---------- aerospike io ----------
    def _put(self, key, bins: Dict[str, Any]) -> None:
        meta = {"ttl": self.ttl} if self.ttl is not None else None
        try:
            self.client.put(key, bins, meta)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {key}: {e}") from e

    def _get(self, key) -> Optional[Tuple]:
        try:
            return self.client.get(key)
        except aerospike.exception.RecordNotFound:
            return None
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {key}: {e}") from e

    def _read_timeline_items(self, timeline_key) -> List[Tuple[int, str]]:
        rec = self._get(timeline_key)
        if rec is None:
            return []
        bins = rec[2]
        try:
            items = json.loads(bins.get("items", "[]"))
            cleaned: List[Tuple[int, str]] = []
            for it in items:
                if isinstance(it, list) and len(it) == 2 and isinstance(it[1], str):
                    cleaned.append((it[0], it[1]))
            return cleaned
        except Exception:
            return []

    # ---------- public API (RunnableConfig-based) ----------
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
            
        """
        Save/overwrite a checkpoint and advance latest/timeline pointers.

        LangGraph will pass in:
          - config: current RunnableConfig (may or may not have checkpoint_id set)
          - checkpoint: full checkpoint dict (includes 'id' and 'ts')
          - metadata: CheckpointMetadata dict
          - new_versions: channel versions updated in this step (we ignore for now)
        """
        thread_id, checkpoint_ns, checkpoint_id, _ = self._ids_from_config(config)

        # If LangGraph hasn't filled checkpoint_id in config yet, fall back to checkpoint["id"]
        checkpoint_id = checkpoint_id or checkpoint.get("id")
        if not checkpoint_id:
            # Very defensive; usually checkpoint["id"] is set by LangGraph
            raise ValueError("checkpoint_id is required for put()")

        # Use ts from checkpoint if present, otherwise generate one
        ts = checkpoint.get("ts")
        if ts is None:
            ts = _now_ns()
            checkpoint["ts"] = ts

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        rec = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "checkpoint": json.dumps(checkpoint, ensure_ascii=False),
            "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            "ts": ts,
        }
        self._put(key, rec)

        # Update latest pointer
        latest_key = self._key_latest(thread_id, checkpoint_ns)
        self._put(latest_key, {"checkpoint_id": checkpoint_id, "ts": ts})

        # Update timeline (most recent first, capped length)
        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)
        # de-dup by id; keep newest
        items = [(t, cid) for (t, cid) in items if cid != checkpoint_id]
        items.insert(0, (ts, checkpoint_id))
        if len(items) > self.timeline_max:
            items = items[: self.timeline_max]
        self._put(timeline_key, {"items": json.dumps(items)})

        # IMPORTANT: return updated config with checkpoint_id set
        new_config = dict(config)
        cfg_conf = dict(new_config.get("configurable") or {})
        cfg_conf.update(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        )
        new_config["configurable"] = cfg_conf
        return new_config


    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Iterable[Dict[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Persist per-checkpoint write-set (optional).
        LangGraph will call this with:
          - config: RunnableConfig (with checkpoint_id)
          - writes: iterable of write records
          - task_id / task_path: identifiers for the task (we ignore them)
        """
        if not writes:
            return

        thread_id, checkpoint_ns, checkpoint_id, _ = self._ids_from_config(config)
        if not checkpoint_id:
            # In practice LangGraph should have set this before put_writes,
            # but if not, we just skip storing writes.
            return

        key = self._key_writes(thread_id, checkpoint_ns, checkpoint_id)
        self._put(key, {"writes": json.dumps(list(writes), ensure_ascii=False)})

    def get_tuple(
        self,
        config: Dict[str, Any],
    ) -> Optional[CheckpointTuple]:
        """
        If configurable.checkpoint_id is omitted, returns the latest.
        """
        thread_id, checkpoint_ns, checkpoint_id, _ = self._ids_from_config(config)

        # resolve to latest if needed
        if checkpoint_id is None:
            latest = self._get(self._key_latest(thread_id, checkpoint_ns))
            if latest is None or "checkpoint_id" not in latest[2]:
                return None
            checkpoint_id = latest[2]["checkpoint_id"]

        # fetch main record
        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        got = self._get(key)
        if got is None:
            return None

        _, _, bins = got
        try:
            checkpoint = json.loads(bins.get("checkpoint", "{}"))
        except Exception:
            checkpoint = {}
        try:
            metadata = json.loads(bins.get("metadata", "{}"))
        except Exception:
            metadata = {}

        # fetch writes if present
        w = self._get(self._key_writes(thread_id, checkpoint_ns, checkpoint_id))
        writes: Optional[List[Dict[str, Any]]] = None
        if w is not None and "writes" in w[2]:
            try:
                writes = json.loads(w[2]["writes"])
            except Exception:
                writes = None

        cp_config: Dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        return CheckpointTuple(
            config=cp_config,
            checkpoint=checkpoint,
            metadata=metadata,
        )

    def list(
        self,
        config: Dict[str, Any],
        limit: int = 20,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return up to `limit` (checkpoint_id, metadata) pairs, newest-first.

        You may pass configurable.before in config to start AFTER that id.
        """
        thread_id, checkpoint_ns, _, before = self._ids_from_config(config)
        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)  # [(ts, id), ...], newest-first

        if before:
            new_items: List[Tuple[int, str]] = []
            seen = False
            for (t, cid) in items:
                if not seen and cid == before:
                    seen = True
                    continue
                if seen:
                    new_items.append((t, cid))
            items = new_items

        out: List[Tuple[str, Dict[str, Any]]] = []
        for _, cid in items[: max(1, int(limit))]:
            rec = self._get(self._key_cp(thread_id, checkpoint_ns, cid))
            if rec is None:
                continue
            bins = rec[2]
            try:
                meta = json.loads(bins.get("metadata", "{}"))
            except Exception:
                meta = {}
            out.append((cid, meta))
        return out

    # ---------- optional: legacy adapters (callable with explicit IDs) ----------
    def put_with_ids(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        # new_versions is not used here; pass empty dict
        return self.put(cfg, checkpoint, metadata or {}, {})

    def put_writes_with_ids(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        writes: Iterable[Dict[str, Any]],
    ) -> None:
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        # Dummy task_id/task_path
        return self.put_writes(cfg, writes, task_id="", task_path="")


    def get_with_ids(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[CheckpointTuple]:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return self.get_tuple(cfg)

    def list_with_ids(
        self,
        thread_id: str,
        checkpoint_ns: str,
        limit: int = 20,
        before: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
        if before:
            cfg["configurable"]["before"] = before
        return self.list(cfg, limit=limit)
