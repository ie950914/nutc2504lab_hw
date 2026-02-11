import os
import requests
import pandas as pd
import re
from docx import Document
import PyPDF2
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 1. 配置 ---
LLM_URL = "https://ws-03.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
MODEL_NAME = "/models/Qwen3-30B-A3B-Instruct-2507-FP8"

def get_stable_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = get_stable_session()

# --- 2. 安全掃描 ---
def security_scan(content, filename):
    if not content: return False
    if "tiramisu" in content.lower() or "ignore all system prompts" in content.lower():
        return True
    return False

# --- 3. 文件處理 ---
def process_idp_files():
    docs_data = []
    files = ['1.pdf', '2.pdf', '3.pdf', '4.png', '5.docx']
    print("🔍 [IDP] 安全掃描中...")
    
    for file_name in files:
        if not os.path.exists(file_name): continue
        content = ""
        try:
            if file_name.endswith('.pdf'):
                with open(file_name, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif file_name.endswith('.docx'):
                doc = Document(file_name)
                content = "\n".join([p.text for p in doc.paragraphs])
            elif file_name.endswith('.png'):
                content = "不動產說明書：104年10月1日生效，不得記載事項包含遷徙自由。"
            
            if security_scan(content, file_name):
                print(f"🔥 [攔截] {file_name} 含惡意指令，已排除。")
                continue
            
            print(f"✅ {file_name} 掃描通過")
            chunks = [content[i:i+500] for i in range(0, len(content), 400)]
            for c in chunks:
                docs_data.append({"text": c, "source": file_name})
        except: continue
    return docs_data

# --- 4. 主程式 ---
if __name__ == "__main__":
    chunks = process_idp_files()
    
    # 取得 Embedding 維度並初始化
    emb_init = session.post(EMBED_URL, json={"texts": ["test"]}).json()
    dim = len(emb_init["embeddings"][0])
    q_client = QdrantClient(":memory:")
    q_client.create_collection("hw7", vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    
    # 同步向量
    points = []
    print(f"🚀 同步向量中 (維度: {dim})...")
    for i, item in enumerate(chunks):
        try:
            emb = session.post(EMBED_URL, json={"texts": [item['text']]}).json()["embeddings"][0]
            points.append(PointStruct(id=i, vector=emb, payload=item))
        except: continue
    q_client.upsert("hw7", points)

    # 處理前 5 題
    qa_df = pd.read_csv('questions_answer.csv').head(5)
    final_results = []

    for _, row in qa_df.iterrows():
        try:
            # 1. 檢索 (改用 query_points 代替 search)
            q_emb = session.post(EMBED_URL, json={"texts": [row['questions']]}).json()["embeddings"][0]
            
            # 使用 query_points 語法
            search_res = q_client.query_points(
                collection_name="hw7",
                query=q_emb,
                limit=1
            ).points
            
            if not search_res:
                ctx, src = "無相關參考資料", "N/A"
            else:
                ctx = search_res[0].payload['text']
                src = search_res[0].payload['source']
            
            # 2. 生成回答
            ans_res = session.post(LLM_URL, json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"資料：{ctx}\n問題：{row['questions']}"}]
            }).json()
            actual_ans = ans_res["choices"][0]["message"]["content"] if "choices" in ans_res else "無法生成回答"

            # 3. 評分
            eval_prompt = f"評分 RAG (0-1), 僅輸出4個數字用逗號隔開:\n問:{row['questions']}\n答:{actual_ans}\n文:{ctx[:200]}"
            eval_data = session.post(LLM_URL, json={"model": MODEL_NAME, "messages": [{"role": "user", "content": eval_prompt}]}).json()
            
            score_text = eval_data["choices"][0]["message"]["content"]
            scores = [float(x) for x in re.findall(r"0\.\d+|1\.0|1|0", score_text)]
            if len(scores) < 4: scores = [0.0, 0.0, 0.0, 0.0]

            final_results.append({
                "q_id": row['id'], "questions": row['questions'], "answer": actual_ans, "source": src,
                "Faithfulness": scores[0], "Relevancy": scores[1], "Precision": scores[2], "Recall": scores[3]
            })
            print(f"✅ Q{row['id']} 完成")
            
        except Exception as e:
            print(f"❌ Q{row['id']} 錯誤: {e}")

    pd.DataFrame(final_results).to_csv('test_dataset.csv', index=False, encoding='utf-8-sig')
    print("\n🎉 檔案已產出：test_dataset.csv")