import aerospike, json

HOST, PORT = "127.0.0.1", 3000
NS = "test"
SET_LATEST = "lg_latest"
SET_CKPT = "lg_checkpoints"

def list_all_latest(limit=50):
    c = aerospike.client({"hosts":[(HOST, PORT)]}).connect()
    try:
        scan = c.scan(NS, SET_LATEST)
        recs = scan.results()  # [(key, meta, bins), ...]
        print(f"Found {len(recs)} latest pointers:")
        rows = []
        for (key, meta, bins) in recs[:limit]:
            rows.append({
                "thread_id": bins.get("thread_id"),
                "checkpoint_ns": bins.get("checkpoint_ns"),
                "checkpoint_id": bins.get("checkpoint_id"),
            })
        for r in rows:
            print(f"  thread={r['thread_id']!r}  ns={r['checkpoint_ns']!r}  ckpt_id={r['checkpoint_id']!r}")
        return rows
    finally:
        c.close()

def show_checkpoint(thread, ns, ckpt_id):
    c = aerospike.client({"hosts":[(HOST, PORT)]}).connect()
    try:
        pk = f"{thread}|{ns}|{ckpt_id}"
        _, meta, bins = c.get((NS, SET_CKPT, pk))
        state = json.loads(bins["state_blob"])
        print("\nCheckpoint:")
        print("  PK:", pk)
        print("  meta:", meta)
        print("  created_at:", bins.get("created_at"))
        print("  state.v:", state.get("v"))
        print("  state (abbrev):", {k: state[k] for k in list(state)[:5]})
    finally:
        c.close()

if __name__ == "__main__":
    rows = list_all_latest()
    if rows:
        # pick the first one
        r = rows[0]
        if r["thread_id"] and r["checkpoint_ns"] and r["checkpoint_id"]:
            show_checkpoint(r["thread_id"], r["checkpoint_ns"], r["checkpoint_id"])
        else:
            print("\nCould not derive PK from bins (missing fields). Try another row or rerun your app.")
