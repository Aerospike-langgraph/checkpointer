# pip install aerospike
import aerospike

# If you're running via docker-compose, host is the service name "aerospike".
# If you published ports to your machine, use "localhost".
CONFIG = {"hosts": [("localhost", 3000)]}  # or ("localhost", 3000)

client = aerospike.client(CONFIG).connect()
try:
    ns, set_name, pk = "test", "demo", "u1"          # (namespace, set, primary key)
    key = (ns, set_name, pk)

    # CREATE / UPSERT
    client.put(key, {"name": "Jagrut", "age": 25, "skills": ["python", "ml"]})
    print("Wrote record:", key)

    # READ
    _key, meta, bins = client.get(key)
    print("Read back:", bins, "(ttl:", meta.get("ttl"), ")")

    # UPDATE (optional example)
    client.put(key, {"age": 26}, policy={"exists": aerospike.POLICY_EXISTS_UPDATE})
    print("Updated age to 26")

    # DELETE
    client.remove(key)
    print("Deleted:", key)

except aerospike.exception.AerospikeError as e:
    print("Aerospike error:", e)
finally:
    client.close()
