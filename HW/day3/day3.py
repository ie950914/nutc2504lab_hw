import os
import time
import requests
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI

# ============================================
# 1. 定義 LangGraph 狀態
# ============================================
class AgentState(TypedDict):
    task_id: str
    raw_txt: str
    raw_srt: str
    detailed_minutes: str
    summary: str

# ============================================
# 2. 定義節點功能
# ============================================

def asr_node(state: AgentState):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    WAV_PATH = os.path.join(base_dir, "audio", "Podcast_EP14.wav") 
    
    if not os.path.exists(WAV_PATH):
        print(f"❌ 找不到音檔：{WAV_PATH}")
        return {"task_id": "ERROR", "raw_txt": "", "raw_srt": ""}

    BASE = "https://3090api.huannago.com"
    CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
    auth = ("nutc2504", "nutc2504")

    print(f"\n📡 [系統] 正在上傳音檔：{os.path.basename(WAV_PATH)}...")
    try:
        with open(WAV_PATH, "rb") as f:
            r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
        r.raise_for_status()
        task_id = r.json()["id"]
    except Exception as e:
        print(f"❌ ASR 上傳失敗: {e}")
        return {"task_id": "ERROR", "raw_txt": "", "raw_srt": ""}

    txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT" 
    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    def wait_download(url: str):
        for _ in range(600):
            try:
                resp = requests.get(url, timeout=(5, 60), auth=auth)
                if resp.status_code == 200: return resp.text
            except: pass
            time.sleep(2)
        return ""

    print(f"⏳ [進度] 任務 ID: {task_id}，語音分析中，請稍候...")
    txt_content = wait_download(txt_url)
    srt_content = wait_download(srt_url)

    return {"task_id": task_id, "raw_txt": txt_content, "raw_srt": srt_content}

def minutes_taker_node(state: AgentState):
    """美化詳細逐字稿"""
    print("🖋️  [處理] 正在格式化詳細逐字稿...")
    lines = state["raw_srt"].split('\n')
    formatted_lines = []
    for line in lines:
        if '-->' in line: # 處理時間軸
            formatted_lines.append(f"\n[🕒 {line.strip()}]")
        elif line.strip().isdigit() or not line.strip(): # 略過序號與空行
            continue
        else: # 內容
            formatted_lines.append(f"  🗣️  {line.strip()}")
            
    header = "┏" + "━"*70 + "┓\n"
    header += "┃" + " "*28 + "📜 詳細會議逐字稿" + " "*28 + "┃\n"
    header += "┗" + "━"*70 + "┛\n"
    
    return {"detailed_minutes": header + "\n".join(formatted_lines)}

def summarizer_node(state: AgentState):
    """美化重點摘要"""
    print("🧠 [處理] 正在生成 AI 重點摘要...")
    client_llm = OpenAI(base_url="https://ws-03.wade0426.me/v1", api_key="EMPTY")
    
    try:
        response = client_llm.chat.completions.create(
            model="/models/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "你是一位專業秘書。請用簡潔的「條列式」整理這段逐字稿的 3 到 5 個關鍵重點。"},
                {"role": "user", "content": state["raw_txt"]}
            ]
        )
        summary_text = response.choices[0].message.content
    except:
        summary_text = "⚠️ 摘要生成暫時失效"

    header = "\n┏" + "━"*70 + "┓\n"
    header += "┃" + " "*28 + "💡 會議重點摘要" + " "*30 + "┃\n"
    header += "┗" + "━"*70 + "┛\n"
    
    return {"summary": header + summary_text}

def writer_node(state: AgentState):
    # 逐字稿區塊
    print(state['detailed_minutes'])
    
    # 分隔線
    print("\n" + "─" * 72)
    
    # 摘要區塊
    print(state['summary'])
    print("  ✅ 任務圓滿完成  ")
    return state

# ============================================
# 3. 構建圖結構 (LangGraph)
# ============================================

workflow = StateGraph(AgentState)
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("asr")
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"task_id": "", "raw_txt": "", "raw_srt": "", "detailed_minutes": "", "summary": ""})