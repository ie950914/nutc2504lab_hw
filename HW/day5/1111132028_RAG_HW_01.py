import os
import pandas as pd
import requests
import json
import re
import numpy as np
import time
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

# 切塊參數
chunk_size = 500
chunk_overlap = 250

# 語意切塊參數（可調整）
SEMANTIC_THRESHOLD = 0.5  # 0.3=切很細, 0.5=中等, 0.7=切很粗

# ============================================
# 工具函數
# ============================================

def get_embeddings(texts, max_retries=3):
    """呼叫 API 取得 embeddings，帶重試機制"""
    for attempt in range(max_retries):
        try:
            res = requests.post(API_EMBED_URL, json={"texts": texts, "normalize": True}, timeout=30)
            if res.status_code == 200:
                return res.json()['embeddings']
            else:
                print(f"⚠️  API 回傳 {res.status_code}, 重試 {attempt+1}/{max_retries}...")
                time.sleep(1)
        except Exception as e:
            print(f"❌ API 呼叫錯誤: {e}, 重試 {attempt+1}/{max_retries}...")
            time.sleep(1)
    
    print(f"❌ API 呼叫失敗，已重試 {max_retries} 次")
    return None

def submit_homework_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        payload["student_answer"] = answer[:2000]
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
    if not vecs:
        print(f"❌ 無法為集合 {name} 建立 embeddings")
        return
    points = [PointStruct(id=i, vector=vecs[i], payload=payloads[i]) for i in range(len(chunks))]
    client.upsert(collection_name=name, points=points)

def semantic_chunking(text, threshold=0.5):
    """真正的語意切塊實作"""
    # 1. 按標點符號切成句子
    sentences = re.split(r'([。！？\n]+)', text)
    sentences = [''.join(sentences[i:i+2]).strip() for i in range(0, len(sentences)-1, 2) if i+1 < len(sentences)]
    sentences = [s for s in sentences if len(s) > 5]
    
    if len(sentences) == 0:
        return [text]
    if len(sentences) == 1:
        return sentences
    
    # 2. 計算每個句子的 embedding
    embeddings = get_embeddings(sentences)
    
    if not embeddings:
        # 回退策略：簡單切塊
        chunks = []
        for i in range(0, len(text), 500):
            chunks.append(text[i:i+500])
        return chunks
    
    embeddings = np.array(embeddings)
    
    # 3. 計算相鄰句子的餘弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i+1]))
        similarities.append(sim)
    
    if not similarities:
        return [text]
    
    # 4. 找出相似度低於閾值的切分點
    split_points = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.append(i + 1)
    
    # 5. 根據切分點組合句子成區塊
    chunks = []
    start = 0
    for split_point in split_points:
        chunk = ''.join(sentences[start:split_point])
        if chunk.strip():
            chunks.append(chunk)
        start = split_point
    
    last_chunk = ''.join(sentences[start:])
    if last_chunk.strip():
        chunks.append(last_chunk)
    
    return chunks if chunks else [text]

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

    df = pd.read_csv(questions_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'questions': 'question', 'question_id': 'q_id', 'id': 'q_id'})

    all_results = []
    
    custom_emb = CustomEmbeddings()
    
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    sliding_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    method_map = {
        "固定大小": ("fixed", fixed_splitter),
        "滑動視窗": ("sliding", sliding_splitter),
        "語意切塊": ("semantic", None)
    }

    print(f"✨ 開始執行 RAG 作業流程 ✨")
    print("=" * 60)

    for m_name, (m_type, splitter) in method_map.items():
        print(f"🚀 正在執行方法：{m_name} ...")
        all_chunks, all_payloads = [], []
        
        for f_path in data_files:
            if not os.path.exists(f_path): continue
            with open(f_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                if m_type == "semantic":
                    chunks = semantic_chunking(content, threshold=SEMANTIC_THRESHOLD)
                else:
                    chunks = splitter.split_text(content)
                
                all_chunks.extend(chunks)
                for c in chunks:
                    all_payloads.append({"text": c, "source": os.path.basename(f_path)})

        print(f"   📦 {m_name} 總共切出 {len(all_chunks)} 個區塊")
        coll_name = f"hw5_{m_name.encode('utf-8').hex()}"
        setup_collection(coll_name, all_chunks, all_payloads)

        method_score = 0
        q_count = 0
        for _, row in df.iterrows():
            q_text = str(row['question'])
            q_id = row['q_id']
            
            q_vec_result = get_embeddings([q_text])
            if not q_vec_result:
                print(f"  ❌ Q{q_id}: embedding 失敗，跳過")
                continue
            
            q_vec = q_vec_result[0]
            
            try:
                search_res = client.query_points(collection_name=coll_name, query=q_vec, limit=3).points
            except Exception as e:
                print(f"  ❌ Q{q_id}: 搜尋失敗 - {e}")
                continue
            
            if search_res:
                combined_answer = "\n".join([res.payload['text'] for res in search_res])
                score = submit_homework_and_get_score(q_id, combined_answer)
                source = search_res[0].payload['source']
                method_score += score
                q_count += 1
                
                print(f"  🔹 Q{q_id} ({m_name}): 分數 {score:.4f}, 來源 {source}")

                all_results.append({
                    "id": len(all_results) + 1,
                    "q_id": q_id,
                    "method": m_name,
                    "retrieve_text": combined_answer,
                    "score": score,
                    "source": source
                })
        
        avg_score = method_score / q_count if q_count > 0 else 0
        print(f"  ✅ {m_name} 執行完畢，總得分: {method_score:.4f}，平均得分: {avg_score:.4f}\n")

    final_df = pd.DataFrame(all_results)
    final_output = os.path.join(base_dir, "1111132028_RAG_HW_01.csv")
    final_df.to_csv(final_output, index=False, encoding="utf-8-sig")

    print("=" * 60)
    summary = final_df.groupby('method')['score'].agg(['sum', 'mean']).sort_values(by='sum', ascending=False)
    
    for method, stats in summary.iterrows():
        print(f"📊 方法：{method:10} | 總分：{stats['sum']:.4f} | 平均分：{stats['mean']:.4f}")
    
    print("-" * 60)
    best_method = summary.index[0]
    print(f"🏆 表現最好的方法：{best_method}")
    print("=" * 60)

if __name__ == "__main__":
    main()