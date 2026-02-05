import requests
from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ============================================================================
# 1. 配置
# ============================================================================

# LLM 配置
llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY",
    model="/models/gpt-oss-120b",
    temperature=0
)

# 搜尋引擎 URL
SEARXNG_URL = "https://ws-searxng.huannago.com/search"

# 快取
CACHE = {}


# ============================================================================
# 2. 狀態定義
# ============================================================================

class State(TypedDict):
    """Graph 狀態"""
    question: str           # 使用者問題
    knowledge: str          # 收集到的知識
    queries: List[str]      # 搜尋關鍵字歷史
    loop: int              # 迴圈次數（從 1 開始）
    answer: str            # 最終答案
    decision: str          # planner 決策


# ============================================================================
# 3. 工具函數
# ============================================================================

def search_web(query: str, limit: int = 2) -> List[Dict]:
    """執行網路搜尋"""
    print(f"🔍 搜尋: {query}")
    
    params = {"q": query, "format": "json", "language": "zh-TW"}
    
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get('results', [])
        valid = [r for r in results if 'url' in r and 'title' in r]
        print(f"✓ 找到 {len(valid)} 筆結果")
        return valid[:limit]
    except Exception as e:
        print(f"✗ 搜尋失敗: {e}")
        return []


# ============================================================================
# 4. Graph 節點
# ============================================================================

def check_cache(state: State) -> Dict:
    """檢查快取節點"""
    q = state["question"]
    print(f"\n{'='*50}")
    print(f"❓ 問題: {q}")
    print(f"{'='*50}\n")
    
    if q in CACHE:
        print("✓ 快取命中")
        return {"answer": CACHE[q], "knowledge": "[快取]"}
    return {}


def planner(state: State) -> Dict:
    """
    決策節點 - 使用 LLM 判斷資訊是否充足
    
    這裡是關鍵！LLM 會評估目前收集到的資訊是否足以回答問題。
    """
    print(f"\n🧠 [Planner] 評估資訊充足度 (第 {state['loop']} 輪)")
    
    # 限制最大搜尋次數（避免過度搜尋）
    MAX_LOOPS = 2
    if state["loop"] > MAX_LOOPS:  # 注意：因為從 1 開始，所以用 > 而非 >=
        print(f"⚠️ 已達搜尋上限 ({MAX_LOOPS} 次)，強制結束")
        return {"decision": "足夠"}
    
    # 如果沒有知識，一定要搜尋
    if not state["knowledge"]:
        print("→ 知識庫為空，需要搜尋")
        return {"decision": "不足"}
    
    # 使用 LLM 判斷資訊是否充足
    prompt = f"""你是資訊評估專家。請判斷以下資訊是否足以回答使用者的問題。

使用者問題: {state['question']}

目前收集的資訊:
{state['knowledge']}

請評估：這些資訊是否足以完整、準確地回答使用者的問題？

回答格式：
- 如果足夠，只回答「足夠」
- 如果不足，只回答「不足」

你的評估:"""
    
    try:
        print("💭 LLM 評估中...")
        response = llm.invoke(prompt).content.strip()
        print(f"📊 LLM 判斷: {response}")
        
        # 判斷 LLM 的回應
        if "足夠" in response or "足够" in response or "YES" in response.upper():
            return {"decision": "足夠"}
        else:
            return {"decision": "不足"}
            
    except Exception as e:
        print(f"✗ LLM 評估失敗: {e}")
        # 失敗時預設為不足
        return {"decision": "不足"}


def query_gen(state: State) -> Dict:
    """
    關鍵字生成節點 - 使用 LLM 生成搜尋關鍵字
    
    技巧：透過良好的 prompt 來限制搜尋範圍，避免過度搜尋
    """
    print(f"\n✍️ [QueryGen] 生成搜尋關鍵字")
    
    # 這裡是關鍵！使用適當的問題套路來限制過度搜尋
    prompt = f"""你是搜尋關鍵字專家。請根據使用者問題生成一個精準的搜尋關鍵字。

使用者問題: {state['question']}

已搜尋過: {', '.join(state['queries']) if state['queries'] else '無'}

要求：
1. 生成一個最相關的中文或英文關鍵字
2. 關鍵字要簡短（1-5 個詞）
3. 避免與已搜尋的關鍵字重複
4. 專注於問題的核心資訊

直接輸出關鍵字，不要解釋。

關鍵字:"""
    
    try:
        query = llm.invoke(prompt).content.strip()
        # 清理可能的引號
        query = query.strip('"\'「」『』')
        print(f"🔑 生成關鍵字: {query}")
        
        return {
            "queries": state["queries"] + [query],
            "loop": state["loop"] + 1  # Loop 遞增
        }
    except Exception as e:
        print(f"✗ 生成失敗: {e}")
        # 失敗時使用原問題
        return {
            "queries": state["queries"] + [state["question"]],
            "loop": state["loop"] + 1  # Loop 遞增
        }


def search_tool(state: State) -> Dict:
    """
    搜尋工具節點 - 執行網路搜尋並整理結果
    """
    print(f"\n🌐 [SearchTool] 執行搜尋")
    
    if not state["queries"]:
        return {}
    
    # 取最新的搜尋關鍵字
    query = state["queries"][-1]
    results = search_web(query, limit=2)
    
    if not results:
        new_info = f"\n[第 {state['loop']} 次搜尋] 關鍵字「{query}」無結果\n"
    else:
        new_info = f"\n=== 第 {state['loop']} 次搜尋：{query} ===\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("content", "")[:200]  # 限制長度
            
            new_info += f"\n【來源 {i}】{title}\n"
            new_info += f"連結: {url}\n"
            new_info += f"摘要: {snippet}\n"
        new_info += "\n"
    
    print("✓ 知識庫已更新")
    return {"knowledge": state["knowledge"] + new_info}


def final_answer(state: State) -> Dict:
    """
    最終回答節點 - 根據收集的資訊生成答案
    """
    print(f"\n📝 [FinalAnswer] 生成答案")
    
    if not state["knowledge"] or "[快取]" in state["knowledge"]:
        # 快取直接返回
        return {}
    
    prompt = f"""請根據以下資訊，以繁體中文回答使用者的問題。

使用者問題: {state['question']}

收集到的資訊:
{state['knowledge']}

要求：
1. 直接回答問題的核心
2. 引用具體的資訊來源
3. 簡潔清晰，條理分明
4. 如果資訊不完整，請誠實說明

回答:"""
    
    try:
        answer = llm.invoke(prompt).content
        print("✓ 答案生成完成")
        
        # 存入快取
        CACHE[state["question"]] = answer
        
        return {"answer": answer}
    except Exception as e:
        print(f"✗ 生成失敗: {e}")
        return {"answer": f"生成答案時發生錯誤: {e}"}


# ============================================================================
# 5. 構建 Graph
# ============================================================================

def build_graph():
    """構建工作流程圖"""
    
    workflow = StateGraph(State)
    
    # 添加節點
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("planner", planner)
    workflow.add_node("query_gen", query_gen)
    workflow.add_node("search_tool", search_tool)
    workflow.add_node("final_answer", final_answer)
    
    # 設置入口
    workflow.set_entry_point("check_cache")
    
    # 條件路由 - 快取檢查
    def cache_router(state: State):
        return "結束" if state.get("answer") else "planner"
    
    workflow.add_conditional_edges(
        "check_cache",
        cache_router,
        {"結束": END, "planner": "planner"}
    )
    
    # 條件路由 - 決策
    def plan_router(state: State):
        return "final_answer" if state.get("decision") == "足夠" else "query_gen"
    
    workflow.add_conditional_edges(
        "planner",
        plan_router,
        {"final_answer": "final_answer", "query_gen": "query_gen"}
    )
    
    # 固定邊
    workflow.add_edge("query_gen", "search_tool")
    workflow.add_edge("search_tool", "planner")
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()


# ============================================================================
# 6. 主程式
# ============================================================================

def main():
    """主程式入口"""
    
    print("="*60)
    print("🤖 自動查證 AI")
    print("="*60)
    
    # 構建 graph
    app = build_graph()
    
    # 單次執行模式
    print("\n請輸入您想查詢的問題:")
    question = input("❓ 您的問題: ").strip()
    
    if not question:
        print("⚠️ 未輸入問題，程式結束")
        return
    
    # 初始狀態（loop 從 1 開始）
    initial_state = {
        "question": question,
        "knowledge": "",
        "queries": [],
        "loop": 1,  # 從 1 開始計數
        "answer": "",
        "decision": ""
    }
    
    # 執行工作流
    try:
        for output in app.stream(initial_state):
            pass  # 節點內部已有 print，這裡不需要額外輸出
        
        # 顯示結果
        if question in CACHE:
            print(f"\n{'='*60}")
            print("📄 最終答案:")
            print(f"{'='*60}")
            print(CACHE[question])
            print(f"{'='*60}\n")
        else:
            print("\n❌ 未能生成答案\n")
            
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}\n")


if __name__ == "__main__":
    main()