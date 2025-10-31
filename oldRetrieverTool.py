import os
import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from mcp.server.fastmcp import FastMCP

# -------------------------------------------------------------
# MCP Server: APIKnowledgeBaseRetriever
# Tool: query_api_knowledge_base
# -------------------------------------------------------------
# INPUT:
# { "query": "How to get total crop quantity for a state?" }
#
# OUTPUT:
# {
#   "answer": "...",
#   "sources": ["API Name", "Endpoint"],
#   "error": null | { "message": str, "type": str }
# }
# -------------------------------------------------------------

mcp = FastMCP("APIKnowledgeBaseRetriever")

# ---- Step 1: Load Knowledge Base ----
KB_PATH = "kb.json"  # Save your KB JSON file here

try:
    with open(KB_PATH, "r") as f:
        KB = json.load(f)
    print(f"✅ Loaded {len(KB)} API records from KB.")
except Exception as e:
    print(f"❌ Failed to load KB file: {e}")
    KB = []

# ---- Step 2: Build LlamaIndex Documents ----
docs = []
for item in KB:
    pre_api_text = ""
    if item.get("preExicutionRequriedApi"):
        pre_api_text = "\nPre-Execution Required APIs:\n" + json.dumps(item["preExicutionRequriedApi"], indent=2)

    text = f"""
    API Name: {item['name']}
    Description: {item['description']}
    Endpoint: {item['endpoint']}
    Params: {json.dumps(item['params'], indent=2)}
    Response: {json.dumps(item['response'], indent=2)}
    Example URL: {item['example_url']}
    Tags: {', '.join(item.get('tags', []))}
    {pre_api_text}
    """

    docs.append(Document(
        text=text,
        metadata={
            "id": item["id"],
            "endpoint": item["endpoint"],
            "name": item["name"],
            "tags": item.get("tags", []),
            "preExicutionRequriedApi": item.get("preExicutionRequriedApi", [])
        }
    ))

# ---- Step 3: Create Embedding Model + Index ----
try:
    # Use local free embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build the index only using embeddings (no OpenAI required)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    # Set up retriever with top-k similarity search
    retriever = index.as_retriever(similarity_top_k=3)

    # Create query engine without LLM — directly retrieve docs
    #query_engine = RetrieverQueryEngine(retriever=retriever)

    query_engine = retriever
    

    print("✅ KB index built successfully. Server is hot-start ready.")

except Exception as e:
    print(f"❌ Failed to build index: {e}")
    query_engine = None


# ---- Step 4: MCP Tool Definition ----
@mcp.tool()
def query_api_knowledge_base(query: str) -> dict:
    """
    Retrieve the most relevant API endpoints from the KB based on query. The apis are about indian agriculture data like Crop production , SPA details, states producing the specified variety , etc .
        
    """
    if not query_engine:
        return {
            "answer": None,
            "sources": [],
            "error": {"message": "Knowledge base not available", "type": "EngineInitError"}
        }

    try:
        results = retriever.retrieve(query)
        if not results:
            return {"answer": "No relevant API found.", "sources": [], "error": None}

        # Construct a compact but informative answer
        formatted = []
        for r in results:
            meta = r.metadata
            pre_req = meta.get("preExicutionRequriedApi", [])
            pre_req_summary = (
                "\n    ↳ Requires: " + ", ".join(
                    [f"API #{p['apiId']} ({p['inputParam']} ← {p['outputParam']})" for p in pre_req]
                )
            ) if pre_req else ""
            formatted.append(f"🔹 {meta['name']} → {meta['endpoint']} (Score: {r.score:.3f}){pre_req_summary}")

        answer = "\n".join(formatted)
        sources = [r.metadata["endpoint"] for r in results]

        return {"answer": answer, "sources": sources, "error": None}

    except Exception as e:
        return {
            "answer": None,
            "sources": [],
            "error": {"message": str(e), "type": "QueryError"}
        }


# ---- Step 5: Run MCP Server (Streamable HTTP) ----
if __name__ == "__main__":
    print("🌐 Starting MCP server on streamable HTTP...")
    mcp.run(transport="streamable-http")

