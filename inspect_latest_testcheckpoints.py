import aerospike, json

HOST = "127.0.0.1"
PORT = 3000
NS = "test"
SET_LATEST = "lg_latest"
SET_CKPT = "lg_checkpoints"

THREAD = "demo-thread-11"   # <-- set this to the thread you're testing
NSNAME = "default"          # <-- your checkpoint_ns

# If you're using the simple latest|thread key:
KEY_LATEST = (NS, SET_LATEST, f"latest|{THREAD}")
# If you previously used latest|thread|ns, uncomment this and try it as fallback:
KEY_LATEST_LEGACY = (NS, SET_LATEST, f"latest|{THREAD}|{NSNAME}")

client = aerospike.client({"hosts": [(HOST, PORT)]}).connect()

def pretty(obj):
    print(json.dumps(obj, indent=2))

try:
    # 1) Read latest pointer
    try:
        key_used = KEY_LATEST
        _, _, latest_bins = client.get(KEY_LATEST)
        print("✅ Found latest (new shape): PK =", key_used[2])
    except aerospike.exception.RecordNotFound:
        key_used = KEY_LATEST_LEGACY
        _, _, latest_bins = client.get(KEY_LATEST_LEGACY)
        print("✅ Found latest (legacy shape): PK =", key_used[2])

    print("Latest bins:")
    pretty(latest_bins)

    ck_id = latest_bins["checkpoint_id"]
    print("\ncheckpoint_id from latest:", ck_id)

    # 2) Read the checkpoint record that latest points to
    # If your saver now uses thread|ck as key:
    ck_pk_new = f"{THREAD}|{ck_id}"
    key_ck_new = (NS, SET_CKPT, ck_pk_new)

    # If you previously used thread|ns|ck:
    ck_pk_legacy = f"{THREAD}|{NSNAME}|{ck_id}"
    key_ck_legacy = (NS, SET_CKPT, ck_pk_legacy)

    try:
        _, _, ck_bins = client.get(key_ck_new)
        print("\n✅ Found checkpoint (new key): PK =", ck_pk_new)
    except aerospike.exception.RecordNotFound:
        _, _, ck_bins = client.get(key_ck_legacy)
        print("\n✅ Found checkpoint (legacy key): PK =", ck_pk_legacy)

    print("Checkpoint bins (truncated):")
    # Don’t print the whole state_blob yet, just inspect its structure
    print(" thread_id:", ck_bins.get("thread_id"))
    print(" checkpoint_ns:", ck_bins.get("checkpoint_ns"))
    print(" checkpoint_id:", ck_bins.get("checkpoint_id"))

    checkpoint = json.loads(ck_bins["state_blob"])
    print("\nDecoded state_blob:")
    print(" type:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print(" keys:", list(checkpoint)[:10])
        print(" has 'v':", "v" in checkpoint)
    else:
        print(" value:", checkpoint)

except aerospike.exception.RecordNotFound:
    print("❌ Could not find latest or checkpoint for that thread/ns.")
finally:
    client.close()
