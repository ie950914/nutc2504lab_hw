import os
import pandas as pd
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class CustomEmbeddings:
    def embed_documents(self, texts):
        return get_embeddings(texts)
    def embed_query(self, text):
        return get_embeddings([text])[0]

# ============================================
# 配置區
# ============================================
API_EMBED_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"
SERVER_URL = "https://hw-01.wade0426.me/submit_answer"

client = QdrantClient(url=QDRANT_URL)

# ============================================
# 工具函數
# ============================================

def get_embeddings(texts):
    res = requests.post(API_EMBED_URL, json={"texts": texts, "normalize": True})
    return res.json()['embeddings'] if res.status_code == 200 else None

def submit_homework_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SERVER_URL, json=payload)
        return response.json().get('score', 0) if response.status_code == 200 else 0
    except:
        return 0

def setup_collection(name, chunks, payloads):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )
    vecs = get_embeddings(chunks)
    points = [PointStruct(id=i, vector=vecs[i], payload=payloads[i]) for i in range(len(chunks))]
    client.upsert(collection_name=name, points=points)

# ============================================
# 主程式
# ============================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(base_dir, "questions.csv")
    data_files = [os.path.join(base_dir, f"data_0{i}.txt") for i in range(1, 6)]

    if not os.path.exists(questions_path):
        print("❌ 找不到 questions.csv，請確認檔案路徑！")
        return

    # 讀取與對齊欄位
    df = pd.read_csv(questions_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'questions': 'question', 'question_id': 'q_id', 'id': 'q_id'})

    all_results = []
    
    # 參數設定 (同步同學風格設定)
    chunk_size = 500
    chunk_overlap = 250
    
    custom_emb = CustomEmbeddings()
    semantic_splitter = SemanticChunker(custom_emb) 
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    sliding_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    method_map = {
        "固定大小": fixed_splitter,
        "滑動視窗": sliding_splitter,
        "語意切塊": semantic_splitter
    }

    print(f"✨ 開始執行 RAG 作業流程 ✨")
    print("=" * 60)

    for m_name, splitter in method_map.items():
        print(f"🚀 正在執行方法：{m_name} ...")
        all_chunks, all_payloads = [], []
        
        # 1. 執行切塊
        for f_path in data_files:
            if not os.path.exists(f_path): continue
            with open(f_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = splitter.split_text(content)
                all_chunks.extend(chunks)
                for c in chunks:
                    all_payloads.append({"text": c, "source": os.path.basename(f_path)})

        # 2. 存入 VDB
        coll_name = f"hw5_{m_name.encode('utf-8').hex()}"
        setup_collection(coll_name, all_chunks, all_payloads)

        # 3. 逐題檢索與評分 (一個一個跳出來的效果)
        method_score = 0
        for _, row in df.iterrows():
            q_text = str(row['question'])
            q_id = row['q_id']
            
            q_vec = get_embeddings([q_text])[0]
            search_res = client.query_points(collection_name=coll_name, query=q_vec, limit=1).points
            
            if search_res:
                score = submit_homework_and_get_score(q_id, search_res[0].payload['text'])
                source = search_res[0].payload['source']
                method_score += score
                
                # 這裡保留了你喜歡的動態輸出感
                print(f"  🔹 Q{q_id} ({m_name}): 分數 {score:.4f}, 來源 {source}")

                all_results.append({
                    "id": len(all_results) + 1,
                    "q_id": q_id,
                    "method": m_name,
                    "retrieve_text": search_res[0].payload['text'],
                    "score": score,
                    "source": source
                })
        print(f"  ✅ {m_name} 執行完畢，總得分: {method_score:.4f}\n")

    # 4. 存檔與統計
    final_df = pd.DataFrame(all_results)
    final_output = os.path.join(base_dir, "1111132028_RAG_HW_01.csv")
    final_df.to_csv(final_output, index=False, encoding="utf-8-sig")

    print("=" * 60)
    summary = final_df.groupby('method')['score'].sum().sort_values(ascending=False)
    best_method = summary.index[0]
    
    print(f"🏆 表現最好的方法：{best_method} (總分: {summary[best_method]:.4f})")
    print(f"🛠️  參數設定：固定大小({chunk_size}), 滑動視窗({chunk_size}, step {chunk_overlap})")
    print("=" * 60)

if __name__ == "__main__":
    main()