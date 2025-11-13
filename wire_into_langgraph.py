# app.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.aerospike.saver import AerospikeSaver
import aerospike

CONFIG = {"hosts": [("localhost", 3000)]}  # or ("localhost", 3000)

client = aerospike.client(CONFIG).connect()


class State(TypedDict):
    question: str
    answer: str

def answerer(s: State) -> State:
    print(">>> running answerer node")
    s["answer"] = f"Aerospike is running. You asked: {s['question']}"
    return s

g = StateGraph(State)
g.add_node("answerer", answerer)
g.add_edge(START, "answerer")
g.add_edge("answerer", END)

checkpointer = AerospikeSaver(client = client, namespace="test")  # default namespace in CE
app = g.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-10", "checkpoint_ns": "default"}}
    out = app.invoke({"question": "Does this persist?"}, config=config)
    print("RESULT:", out)
