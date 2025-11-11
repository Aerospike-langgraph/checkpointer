import aerospike
client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
print(client.info_all("namespaces"))
client.close()
