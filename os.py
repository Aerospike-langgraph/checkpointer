import os
host = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
port = int(os.getenv("AEROSPIKE_PORT", "3000"))
namespace = os.getenv("AEROSPIKE_NAMESPACE", "lg")
