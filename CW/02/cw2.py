import requests
import pandas as pd
import re
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================
# 設定與初始化
# ============================================

API_EMBED_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"


# 連接 Qdrant
try:
    client = QdrantClient(url=QDRANT_URL)
    print("✅ 已成功連接至 Qdrant VDB")
except Exception as e:
    print(f"❌ 無法連接 Qdrant: {e}")
    exit()

# ============================================
# 工具函數
# ============================================

def get_embeddings(texts):
    """取得文本向量"""
    response = requests.post(API_EMBED_URL, json={
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    })
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ 向量生成失敗: {response.status_code}")
        return None

def markdown_to_csv(md_file, csv_file):
    """Markdown 表格轉 CSV"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    data = []

    for line in lines:
        # 跳過表格分隔線
        if '|' not in line or line.strip() == '':
            continue
        # 跳過只有分隔符的行
        if all(c in '|-: \t' for c in line.replace('|', '')):
            continue
        
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]
        if cells:
            data.append(cells)

    if len(data) > 1:
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        return df
    else:
        return None

# ============================================
# 第一部分：文本切塊與嵌入
# ============================================

print("\n" + "="*60)
print("第一部分：文本切塊處理")
print("="*60)

# 讀取文本文件
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"📄 原始文本長度: {len(text)} 字符")

# 1. 固定切塊 (無重疊)
print("\n【固定切塊】")
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

fixed_chunks = fixed_splitter.split_text(text)
print(f"✅ 固定切塊產生 {len(fixed_chunks)} 個方塊")

# 2. 滑動視窗切塊 (有重疊)
print("\n【滑動視窗切塊】")
sliding_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

sliding_chunks = sliding_splitter.split_text(text)
print(f"✅ 滑動視窗切塊產生 {len(sliding_chunks)} 個方塊")

# 3. 生成向量嵌入
print("\n📊 開始生成向量嵌入...")

# 固定切塊嵌入
fixed_data = {
    "texts": fixed_chunks,
    "normalize": True,
    "batch_size": 32
}
fixed_response = requests.post(API_EMBED_URL, json=fixed_data)

if fixed_response.status_code == 200:
    fixed_result = fixed_response.json()
    print(f"✅ 固定切塊向量維度: {fixed_result['dimension']}")
else:
    print(f"❌ 固定切塊嵌入失敗")
    exit()

# 滑動視窗切塊嵌入
sliding_data = {
    "texts": sliding_chunks,
    "normalize": True,
    "batch_size": 32
}
sliding_response = requests.post(API_EMBED_URL, json=sliding_data)

if sliding_response.status_code == 200:
    sliding_result = sliding_response.json()
    print(f"✅ 滑動視窗切塊向量維度: {sliding_result['dimension']}")
else:
    print(f"❌ 滑動視窗切塊嵌入失敗")
    exit()

# 4. 存入 Qdrant
print("\n💾 開始存入 Qdrant...")

# 建立固定切塊集合
if client.collection_exists("fixed_collection"):
    client.delete_collection("fixed_collection")

client.create_collection(
    collection_name="fixed_collection",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# 建立滑動視窗切塊集合
if client.collection_exists("sliding_collection"):
    client.delete_collection("sliding_collection")

client.create_collection(
    collection_name="sliding_collection",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# 插入固定切塊向量
fixed_points = []
for i, vec in enumerate(fixed_result['embeddings']):
    fixed_points.append(
        PointStruct(
            id=i + 1,
            vector=vec,
            payload={"text": fixed_chunks[i], "chunk_type": "fixed", "chunk_id": i}
        )
    )

client.upsert(collection_name="fixed_collection", points=fixed_points)
print(f"✅ 成功插入 {len(fixed_points)} 個固定切塊向量")

# 插入滑動視窗切塊向量
sliding_points = []
for i, vec in enumerate(sliding_result['embeddings']):
    sliding_points.append(
        PointStruct(
            id=i + 1,
            vector=vec,
            payload={"text": sliding_chunks[i], "chunk_type": "sliding", "chunk_id": i}
        )
    )

client.upsert(collection_name="sliding_collection", points=sliding_points)
print(f"✅ 成功插入 {len(sliding_points)} 個滑動視窗切塊向量")

# ============================================
# 第二部分：召回測試與比較
# ============================================

print("\n" + "="*60)
print("第二部分：召回測試與比較")
print("="*60)

# 測試問題
test_queries = [
    "Graph RAG 有什麼優勢?",
    "知識圖譜如何建構?",
    "微軟 GraphRAG 的特點是什麼?"
]

for query_text in test_queries:
    print(f"\n🔍 查詢問題: {query_text}")
    print("-"*60)
    
    # 生成查詢向量
    query_data = {
        "texts": [query_text],
        "normalize": True,
        "batch_size": 32
    }
    query_response = requests.post(API_EMBED_URL, json=query_data)
    
    if query_response.status_code != 200:
        print("❌ 查詢向量生成失敗")
        continue
    
    query_vector = query_response.json()['embeddings'][0]
    
    # 固定切塊查詢
    fixed_search = client.query_points(
        collection_name="fixed_collection",
        query=query_vector,
        limit=3
    )
    
    # 滑動視窗切塊查詢
    sliding_search = client.query_points(
        collection_name="sliding_collection",
        query=query_vector,
        limit=3
    )
    
    # 比較最高分數
    fixed_max_score = max([p.score for p in fixed_search.points]) if fixed_search.points else 0
    sliding_max_score = max([p.score for p in sliding_search.points]) if sliding_search.points else 0
    
    print(f"\n📊 固定切塊最高分: {fixed_max_score:.4f}")
    print(f"   最佳結果: {fixed_search.points[0].payload['text'][:80]}...")
    
    print(f"\n📊 滑動視窗最高分: {sliding_max_score:.4f}")
    print(f"   最佳結果: {sliding_search.points[0].payload['text'][:80]}...")
    
    winner = "滑動視窗" if sliding_max_score > fixed_max_score else "固定切塊"
    print(f"\n🏆 本次查詢獲勝: {winner}")

# ============================================
# 第三部分：表格處理
# ============================================

print("\n" + "="*60)
print("第三部分：表格處理")
print("="*60)

# 1. Markdown 表格轉 CSV
print("\n【處理 Markdown 表格】")
md_df = markdown_to_csv('table_txt.md', 'output.csv')
if md_df is not None:
    print(f"✅ Markdown 表格已轉換為 output.csv")
    print(f"   表格大小: {len(md_df)} 行 x {len(md_df.columns)} 列")

# 2. 讀取 HTML 表格
print("\n【處理 HTML 表格】")
try:
    tables = pd.read_html("table_html.html", encoding="UTF-8")
    print(f"✅ 從 HTML 中讀取到 {len(tables)} 個表格")
    print(f"   表格大小: {tables[0].shape[0]} 行 x {tables[0].shape[1]} 列")
except Exception as e:
    print(f"❌ 讀取 HTML 表格失敗: {e}")
    tables = None

# 3. 使用 LLM 生成表格摘要 (Prompt v1)
if tables is not None:
    print("\n" + "="*60)
    print("使用 LLM 生成表格摘要 (Prompt v1)...")
    print("="*60)
    
    with open("Prompt_table_v1.txt", "r", encoding="UTF-8") as f:
        system_prompt_v1 = f.read()
    
    client_llm = OpenAI(
        base_url="https://ws-03.wade0426.me/v1",
        api_key="EMPTY",
    )
    
    response_v1 = client_llm.chat.completions.create(
        model="/models/gpt-oss-120b",
        messages=[
            {"role": "system", "content": f"{system_prompt_v1}"},
            {"role": "user", "content": f"{tables[0].to_string()}"}
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
        stream=True
    )
    
    print("\n生成的摘要:")
    print("-"*60)
    table_summary = ""
    for chunk in response_v1:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            table_summary += content
    
    print("\n" + "-"*60)
    
    # 4. 使用 LLM 生成問答對 (Prompt v2)
    print("\n" + "="*60)
    print("使用 LLM 生成問答對 (Prompt v2)...")
    print("="*60)
    
    with open("Prompt_table_v2.txt", "r", encoding="UTF-8") as f:
        system_prompt_v2 = f.read()
    
    response_v2 = client_llm.chat.completions.create(
        model="/models/gpt-oss-120b",
        messages=[
            {"role": "system", "content": f"{system_prompt_v2}"},
            {"role": "user", "content": f"{tables[0].to_string()}"}
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
        stream=False
    )
    
    qa_json = response_v2.choices[0].message.content
    print("\n生成的問答對:")
    print("-"*60)
    print(qa_json)
    print("-"*60)
    
    # 5. 將表格摘要和問答對存入 Qdrant
    print("\n💾 開始將表格資料存入 Qdrant...")
    
    # 建立表格集合
    if client.collection_exists("table_collection"):
        client.delete_collection("table_collection")
    
    client.create_collection(
        collection_name="table_collection",
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )
    
    # 準備所有文本（摘要 + 問答對）
    all_table_texts = [table_summary]
    
    # 解析問答對
    try:
        import json
        qa_list = json.loads(qa_json)
        for qa in qa_list:
            qa_text = f"問題: {qa['question']}\n答案: {qa['answer']}"
            all_table_texts.append(qa_text)
        print(f"✅ 成功解析 {len(qa_list)} 組問答對")
    except:
        print("⚠️  問答對格式解析失敗，僅保留摘要")
    
    # 生成向量
    table_embed_data = {
        "texts": all_table_texts,
        "normalize": True,
        "batch_size": 32
    }
    table_embed_response = requests.post(API_EMBED_URL, json=table_embed_data)
    
    if table_embed_response.status_code == 200:
        table_embed_result = table_embed_response.json()
        
        # 存入 Qdrant
        table_points = []
        for i, vec in enumerate(table_embed_result['embeddings']):
            point_type = "table_summary" if i == 0 else "table_qa"
            table_points.append(
                PointStruct(
                    id=i + 1,
                    vector=vec,
                    payload={
                        "text": all_table_texts[i],
                        "type": point_type,
                        "source": "table_html.html"
                    }
                )
            )
        
        client.upsert(collection_name="table_collection", points=table_points)
        print(f"✅ 成功上傳 {len(table_points)} 個表格相關資料到 Qdrant")
        print(f"   - 1 個表格摘要")
        print(f"   - {len(table_points)-1} 個問答對")
    else:
        print("❌ 表格向量生成失敗")

# ============================================
# 第四部分：表格查詢測試
# ============================================

print("\n" + "="*60)
print("第四部分：表格查詢測試")
print("="*60)

table_queries = ["台中科大有什麼特色?", "學校的發展計畫是什麼?"]

for query_text in table_queries:
    print(f"\n🔍 查詢問題: {query_text}")
    print("-"*60)
    
    query_data = {
        "texts": [query_text],
        "normalize": True,
        "batch_size": 32
    }
    query_response = requests.post(API_EMBED_URL, json=query_data)
    
    if query_response.status_code == 200:
        query_vector = query_response.json()['embeddings'][0]
        
        search_result = client.query_points(
            collection_name="table_collection",
            query=query_vector,
            limit=3
        )
        
        for idx, point in enumerate(search_result.points, 1):
            print(f"\n結果 {idx}:")
            print(f"  類型: {point.payload['type']}")
            print(f"  相似度: {point.score:.4f}")
            print(f"  內容: {point.payload['text'][:150]}...")

