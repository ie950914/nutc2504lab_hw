import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, Range

data = {
    "texts": [
        "人工智慧很有趣",
        "知識圖譜RAG技術介紹與探討", 
        "透過專有知識庫輔助生成過程", 
        "以及生成高階實體與摘要",
        "適合需細緻控制查詢邏輯的場景及搭配客製化設計的推理機制技術劣勢"
    ],
    "normalize": True,
    "batch_size": 32
}
response = requests.post("https://ws-04.wade0426.me/embed", json=data)

print(f"狀態碼:{response.status_code}")
print(f"回應內容:{response.text}")

if response.status_code == 200:
    result = response.json()
    print(f"維度:{result['dimension']}")
else:
    print(f"錯誤:{response.json()}")
    exit()

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists("test_collection"):
    client.delete_collection("test_collection")

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

points = []
for i, vec in enumerate(result['embeddings']):
    points.append(
        PointStruct(
            id=i + 1,
            vector=vec,
            payload={"text": data["texts"][i], "metadata": "其他資訊", "year": 5}
        )
    )

client.upsert(
    collection_name="test_collection",
    points=points
)

texts = ["AI 有什麼好處?"]
query_data = {
    "texts": texts,
    "normalize": True,
    "batch_size": 32
}
query_response = requests.post("https://ws-04.wade0426.me/embed", json=query_data)
if query_response.status_code == 200:
    query_result = query_response.json()
    query_vector = query_result['embeddings'][0] # 修正：加 s
else:
    print("查詢向量取得失敗")
    exit()

search_result = client.query_points(
    collection_name="test_collection",
    query=query_vector,
    limit=3
)
for point in search_result.points:
    print(f"ID:{point.id}")
    print(f"相似度分數(Score):{point.score}")
    print(f"內容:{point.payload['text']}")
    print("---")

result_filter = client.query_points(
    collection_name="test_collection",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key='year',
                range=Range(
                    gte=3,
                    lte=10
                )
            )
        ]
    ),
    limit=5
)
for point in result_filter.points:
    print(f"ID:{point.id}")
    print(f"Payload:{point.payload}")
    print("-" * 30)