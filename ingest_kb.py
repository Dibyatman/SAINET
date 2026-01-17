import json
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

KB_PATH = "kb.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "api_kb"

with open(KB_PATH, "r") as f:
    KB = json.load(f)

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
            text=text.strip(),
            metadata={
                "id": int(item["id"]),
                "endpoint": item["endpoint"],
                "name": item["name"],
                "tags": ",".join(item.get("tags", [])),  # ✅ FLATTENED
                "preExicutionRequriedApi": json.dumps(   # ✅ SERIALIZED
                    item.get("preExicutionRequriedApi", [])
                ),
            },
        )
    )

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("✅ KB ingested successfully into persistent ChromaDB")
