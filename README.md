# LangGraph Aerospike Checkpointer

Store LangGraph checkpoints in Aerospike using the provided `AerospikeSaver`. The repo includes a minimal Aerospike docker setup, examples, and pytest-based checks.

## Quick start
1) Install deps (Python 3.10+/aerospike client):  
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```
2) Bring up Aerospike locally:  
   ```bash
   docker compose up -d
   ```
3) Point the saver at your cluster (defaults work for the docker compose):  
   - `AEROSPIKE_HOST=127.0.0.1`  
   - `AEROSPIKE_PORT=3000`  
   - `AEROSPIKE_NAMESPACE=test`
4) Use in a graph:
   ```python
   import aerospike
   from langgraph.checkpoint.aerospike import AerospikeSaver

   client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
   saver = AerospikeSaver(client=client, namespace="test")

   compiled = graph.compile(checkpointer=saver)  # graph is your LangGraph graph
   compiled.invoke({"input": "hello"}, config={"configurable": {"thread_id": "demo"}})
   ```

## Configuration
- The saver reads `AEROSPIKE_HOST`, `AEROSPIKE_PORT`, and `AEROSPIKE_NAMESPACE`; defaults match the compose file.
- Create a `.env` if you want to override locally (not committed).
- For custom Aerospike configs, edit `docker-compose.yml` (volume mounts are commented as examples).

## Tests
You need a reachable Aerospike instance (compose is fine):
```bash
docker compose up -d
pytest
```

## Utilities
- `inspect_as.py` / `inspect_latest_testcheckpoints.py`: quick helpers to inspect stored checkpoints.
- `wire_into_langgraph.py`: shows how to attach the saver to a LangGraph graph.

## Notes
- Ignore/leave untracked local DBs (`travel2.sqlite`), caches, and virtualenvs; `.gitignore` already covers them.
- If you see tracked `__pycache__` files from earlier commits, run `git rm --cached -r __pycache__ .pytest_cache` once and commit.***
