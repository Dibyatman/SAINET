from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server.fastmcp import FastMCP
import chromadb

mcp = FastMCP("APIKnowledgeBaseRetriever")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "api_kb"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=2)
    query_engine = retriever

    print("✅ Persistent KB loaded. No re-indexing needed.")

except Exception as e:
    print(f"❌ Failed to load persistent KB: {e}")
    query_engine = None


@mcp.tool()
def query_api_knowledge_base(query: str) -> dict:
    """
    Retrieve full API documents from persistent vector DB.
    """
    if not query_engine:
        return {
            "documents": [],
            "sources": [],
            "error": {"message": "KB not available", "type": "InitError"},
        }

    try:
        results = query_engine.retrieve(query)
        if not results:
            return {"documents": [], "sources": [], "error": None}

        documents = []
        for r in results:
            documents.append({
                "text": r.text.strip(),
                "metadata": r.metadata,
                "score": round(r.score, 4),
            })

        sources = [r.metadata.get("endpoint", "") for r in results]

        return {
            "documents": documents,
            "sources": sources,
            "error": None
        }

    except Exception as e:
        return {
            "documents": [],
            "sources": [],
            "error": {"message": str(e), "type": "QueryError"},
        }


if __name__ == "__main__":
    print("🌐 Starting MCP server with persistent vector DB...")
    mcp.run(transport="streamable-http")
