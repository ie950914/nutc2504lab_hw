import os
import csv
import uuid
import torch
import requests
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

# 強制禁用連線，確保讀取本地模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# --- 配置與路徑 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-03.wade0426.me/v1/chat/completions"
LLM_MODEL = "/models/gpt-oss-120b"
RERANKER_PATH = os.path.expanduser("~/AI/Models/Qwen3-Reranker-0.6B")
COLLECTION_NAME = "CW_04_Hybrid_Final"

# --- 1. 載入模型 ---
print("⌛ 正在載入 Reranker 模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True, trust_remote_code=True, use_fast=False)
    # 修正警告：使用 dtype 代替 torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        RERANKER_PATH, local_files_only=True, trust_remote_code=True, 
        dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ 模型載入成功，運行於: {device}")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}"); exit(1)

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

def get_embeddings(texts, task="檢索文件"):
    res = requests.post(EMBED_API_URL, json={"texts": texts, "task_description": task, "normalize": True}).json()
    return res.get("embeddings", [])

def call_llm(prompt):
    res = requests.post(LLM_API_URL, json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}).json()
    return res["choices"][0]["message"]["content"].strip()

@torch.no_grad()
def rerank_docs(query, candidates, limit=3):
    """重排評分邏輯"""
    if not candidates: return []
    pairs = [f"<Instruct>: 根據查詢檢索相關的技術文件\n<Query>: {query}\n<Document>: {d}" for d in candidates]
    
    scores = []
    for p in pairs:
        inputs = tokenizer(p, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(model.device)
        logits = model(**inputs).logits[:, -1, :]
        batch_scores = torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1)
        prob = torch.nn.functional.softmax(batch_scores, dim=1)[:, 1].item()
        scores.append(prob)
    
    combined = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in combined[:limit]]

def main():
    client = QdrantClient("localhost", port=6333)
    
    # 2. 初始化 Hybrid 集合
    sample_emb = get_embeddings(["測試"])
    dim = len(sample_emb[0])
    if client.collection_exists(COLLECTION_NAME): client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )

    # 3. 匯入資料
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for i in range(1, 6):
        path = os.path.join(SCRIPT_DIR, f"data_0{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks = splitter.split_text(f.read())
                embs = get_embeddings(chunks)
                points = [models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={"dense": e, "sparse": models.Document(text=c, model="Qdrant/bm25")},
                    payload={"text": c, "source": f"data_0{i}.txt"}
                ) for c, e in zip(chunks, embs)]
                client.upsert(COLLECTION_NAME, points)

    # 4. 處理問題 (修正欄位為「題目」)
    input_csv = os.path.join(SCRIPT_DIR, "questions.csv")
    with open(input_csv, "r", encoding="utf-8-sig") as f: 
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 根據你的錯誤訊息，這裡手動指定為 '題目'
    q_col = '題目' 

    for idx, r in enumerate(rows, 1):
        user_q = r[q_col].strip()
        q_emb = get_embeddings([user_q], task="查詢")
        
        # Hybrid Search
        search_res = client.query_points(
            COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=models.Document(text=user_q, model="Qdrant/bm25"), using="sparse", limit=15),
                models.Prefetch(query=q_emb[0], using="dense", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15
        ).points
        
        # ReRank
        candidates = [p.payload["text"] for p in search_res]
        top_context = "\n\n".join(rerank_docs(user_q, candidates))
        
        r['answer'] = call_llm(f"資料：\n{top_context}\n\n問題：{user_q}\n請簡潔回答。")
        print(f"[{idx}/{len(rows)}] ✅ 已處理: {user_q[:20]}...")

    # 5. 寫回結果
    out_path = os.path.join(SCRIPT_DIR, "results_04.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        # 修改 fieldnames 以包含我們新增的 'answer' 欄位
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"🎉 全部完成！結果已存至: {out_path}")

if __name__ == "__main__":
    main()