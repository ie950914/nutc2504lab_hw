import os
import csv
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. 配置與路徑設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "gemma-3-27b-it"

COLLECTION_NAME = "CW_03" 
CHUNK_SIZE = 500  # 稍微加大切塊，讓 Context 更完整
CHUNK_OVERLAP = 50

def get_embedding(texts):
    """取得向量與維度"""
    try:
        res = requests.post(EMBED_API_URL, json={
            "texts": texts, "task_description": "檢索文件", "normalize": True
        }, timeout=30).json()
        embs = res.get("embeddings", [])
        return embs, len(embs[0]) if embs else 0
    except Exception as e:
        print(f"❌ Embedding 錯誤: {e}")
        return None, 0

def call_llm(system_prompt, user_prompt):
    """呼叫 LLM API"""
    try:
        res = requests.post(LLM_API_URL, json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }, timeout=60).json()
        return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ LLM 呼叫失敗: {e}")
        return ""

def main():
    # 連接 Qdrant (請確保 sudo docker 已啟動)
    client = QdrantClient("localhost", port=6333)
    
    # --- A. 準備 VDB ---
    print(f"🚀 初始化 VDB: {COLLECTION_NAME}")
    _, dim = get_embedding(["測試"])
    if dim == 0: 
        print("❌ 無法偵測維度，請檢查網路或 API URL"); return
        
    if client.collection_exists(COLLECTION_NAME): 
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    # --- B. 切塊與匯入資料 ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_points = []
    p_idx = 0
    
    # 搜尋同資料夾下的 data_01.txt ~ data_05.txt
    for i in range(1, 6):
        path = os.path.join(SCRIPT_DIR, f"data_0{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = splitter.split_text(content)
                embs, _ = get_embedding(chunks)
                if embs:
                    for c, e in zip(chunks, embs):
                        all_points.append(PointStruct(id=p_idx, vector=e, payload={"text": c, "source": f"data_0{i}.txt"}))
                        p_idx += 1
    
    if all_points:
        client.upsert(COLLECTION_NAME, all_points)
        print(f"✅ 已存入 {p_idx} 個語意塊至 Qdrant")
    else:
        print("❌ 找不到 data_*.txt 檔案，請檢查檔案名稱與位置")

    # --- C. 處理 CSV 問題集 (Query Re-Write 核心) ---
    input_path = os.path.join(SCRIPT_DIR, "Re_Write_questions.csv")
    if not os.path.exists(input_path):
        print(f"❌ 找不到輸入檔: {input_path}"); return

    with open(input_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 按 conversation_id 分組，確保歷史對話邏輯正確
    conv_groups = {}
    for r in rows:
        cid = r['conversation_id']
        if cid not in conv_groups: conv_groups[cid] = []
        conv_groups[cid].append(r)

    final_results = []
    for cid, questions in conv_groups.items():
        history = "" # 每個新 Session 重置對話歷史
        print(f"\n📂 正在處理 Session: {cid}")
        
        for q in questions:
            user_q = q['questions'] # 注意這裡對應 CSV 欄位名稱
            
            # 1. Query Re-Write
            if not history:
                search_query = user_q # 第一題直接搜尋
            else:
                rewrite_sys = "你是一個查詢重寫專家。請根據對話歷史，將使用者的最新問題改寫成一個語意完整且適合搜尋技術文件的獨立句子。嚴禁解釋或廢話。"
                rewrite_usr = f"歷史：{history}\n最新問題：{user_q}\n重寫後的搜尋句："
                search_query = call_llm(rewrite_sys, rewrite_usr).split('\n')[0].replace('"', '')
            
            print(f"   🔎 原始: {user_q[:15]}... -> 搜尋句: {search_query}")

            # 2. 檢索 (Retrieval)
            q_emb, _ = get_embedding([search_query])
            hits = client.query_points(COLLECTION_NAME, query=q_emb[0], limit=3).points
            
            context = "\n".join([h.payload["text"] for h in hits])
            source = hits[0].payload["source"] if hits else "未知"

            # 3. 回答生成 (RAG)
            ans_sys = "你是一個專業的 AI 助手。請根據提供的參考資料，精準且簡短地回答使用者的問題。如果資料中沒有答案，請回答「資料庫無相關記載」。"
            ans_usr = f"【參考資料】：\n{context}\n\n【問題】：{user_q}"
            answer = call_llm(ans_sys, ans_usr)

            # 更新結果與歷史
            q.update({"answer": answer, "source": source})
            final_results.append(q)
            # 簡短紀錄歷史供下次重寫使用
            history += f" Q:{user_q} A:{answer[:10]}"

    # --- D. 寫回結果 ---
    out_path = os.path.join(SCRIPT_DIR, "Re_Write_results.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(final_results)
    
    print(f"\n🎉 處理完成！結果已存至: {out_path}")

if __name__ == "__main__":
    main()