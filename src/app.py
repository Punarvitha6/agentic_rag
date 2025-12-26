from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents import AWSRAGCrew
import uvicorn

# --- LangGraph Orchestration ---
class GraphState(TypedDict):
    query: str
    result: str

def agent_execution_node(state: GraphState):
    crew_service = AWSRAGCrew()
    output = crew_service.run(state['query'])
    return {"result": str(output)}

# Graph Definition
builder = StateGraph(GraphState)
builder.add_node("agent_processing", agent_execution_node)
builder.set_entry_point("agent_processing")
builder.add_edge("agent_processing", END)
app_graph = builder.compile()

# --- FastAPI Microservice ---
app = FastAPI(title="Agentic RAG Assistant Service")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    """
    Check endpoint: Sends query to the Agentic RAG system.
    Body: {"query": "Your question"}
    """
    print(f"📡 API Request Received: {request.query}")
    state = app_graph.invoke({"query": request.query})
    return {
        "status": "success",
        "answer": state["result"]
    }

if __name__ == "__main__":
    # Runs on http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)