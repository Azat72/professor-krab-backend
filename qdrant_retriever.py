import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer

QDRANT_COLLECTION = "legal_documents"

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer

QDRANT_COLLECTION = "legal_documents"
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant-render-cosz.onrender.com")  # твой URL Render Qdrant

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search_qdrant(query: str, top_k: int = 20):
    embedding = model.encode(query).tolist()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )

    # Собираем и группируем результаты по файлам для усиления контекстной связанности
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

    # Сортируем чанки внутри каждой группы по score и разворачиваем список
    sorted_results = []
    for chunks in grouped.values():
        sorted_chunks = sorted(chunks, key=lambda x: -x["score"])
        sorted_results.extend(sorted_chunks)

    return sorted_results
