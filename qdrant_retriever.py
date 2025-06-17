import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from sentence_transformers import SentenceTransformer

# Название коллекции
QDRANT_COLLECTION = "legal_documents"

# Адрес и ключ доступа к Qdrant Cloud
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://62dda2c9-a689-4091-97a3-3ce3f60dcba4.us-east4-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "sk-cdaf10fad862474a94007f3d0d5c66a5"
)

# Инициализация клиента Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Модель эмбеддингов
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search_qdrant(query: str, top_k: int = 20):
    # Векторизация запроса
    embedding = model.encode(query).tolist()

    # Поиск по коллекции
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )

    # Группировка и сортировка результатов по source_file и score
    grouped = {}
    for hit in results:
        file_key = hit.payload.get("source_file", "")
        if file_key not in grouped:
            grouped[file_key] = []
        grouped[file_key].append({
            "text": hit.payload.get("text", ""),
            "source_file": file_key,
            "title": hit.payload.get("title", ""),
            "score": hit.score,
        })

    sorted_results = []
    for chunks in grouped.values():
        sorted_chunks = sorted(chunks, key=lambda x: -x["score"])
        sorted_results.extend(sorted_chunks)

    return sorted_results

