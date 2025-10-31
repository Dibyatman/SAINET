import os
import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
#   "documents": [ { "text": "...", "metadata": {...}, "score": 0.93 }, ... ],
#   "sources": ["API Name", "Endpoint"],
#   "error": null | { "message": str, "type": str }
# }
# -------------------------------------------------------------

mcp = FastMCP("APIKnowledgeBaseRetriever")

# ---- Step 1: Load Knowledge Base ----
KB_PATH = "kb.json"  # Path to your KB JSON file

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
        pre_api_text = "\nPre-Execution Required APIs:\n" + json.dumps(
            item["preExicutionRequriedApi"], indent=2
        )

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

    docs.append(
        Document(
            text=text,
            metadata={
                "id": item["id"],
                "endpoint": item["endpoint"],
                "name": item["name"],
                "tags": item.get("tags", []),
                "preExicutionRequriedApi": item.get("preExicutionRequriedApi", []),
            },
        )
    )

# ---- Step 3: Create Embedding Model + Index ----
try:
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=2)
    query_engine = retriever
    print("✅ KB index built successfully. Server is hot-start ready.")
except Exception as e:
    print(f"❌ Failed to build index: {e}")
    query_engine = None


# ---- Step 4: MCP Tool Definition ----
@mcp.tool()
def query_api_knowledge_base(query: str) -> dict:
    """
    Retrieve full API documents from the knowledge base for .These APIs contain Indian agriculture data such as crop production,SPA details, and state-wise variety information.
    """
    if not query_engine:
        return {
            "documents": [],
            "sources": [],
            "error": {"message": "Knowledge base not available", "type": "EngineInitError"},
        }

    try:
        results = query_engine.retrieve(query)
        if not results:
            return {"documents": [], "sources": [], "error": None}

        # Return full document text, metadata, and score
        docs_out = []
        for r in results:
            docs_out.append(
                {
                    "text": r.text.strip(),
                    "metadata": r.metadata,
                    "score": round(r.score, 4),
                }
            )

        sources = [r.metadata.get("endpoint", "") for r in results]

        return {"documents": docs_out, "sources": sources, "error": None}

    except Exception as e:
        return {
            "documents": [],
            "sources": [],
            "error": {"message": str(e), "type": "QueryError"},
        }


# ---- Step 5: Run MCP Server (Streamable HTTP) ----
if __name__ == "__main__":
    print("🌐 Starting MCP server on streamable HTTP...")
    mcp.run(transport="streamable-http")
