from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from sentence_transformers import SentenceTransformer

# Название коллекции в Qdrant Cloud
QDRANT_COLLECTION = "legal_documents"

# Прямое подключение к твоему Qdrant Cloud
client = QdrantClient(
    url="https://62dda2c9-a689-4091-97a3-3ce3f60dcba4.us-east4-0.gcp.cloud.qdrant.io",
    api_key="sk-cdaf10fad862474a94007f3d0d5c66a5"
)

# Загружаем модель эмбеддингов (размерность 384)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search_qdrant(query: str, top_k: int = 20):
    # Векторизуем запрос
    embedding = model.encode(query).tolist()

    # Выполняем поиск в Qdrant Cloud
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )

    # Группируем результаты по файлам и сортируем по score
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
