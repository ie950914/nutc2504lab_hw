import time
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 模型配置 (依據你的圖片網址設定)
common_kwargs = {"temperature": 0, "api_key": "EMPTY"}
llm_model = ChatOpenAI(base_url="https://ws-03.wade0426.me/v1", model="Llama-3.3-70B-Instruct-NVFP4", **common_kwargs)
vlm_model = ChatOpenAI(base_url="https://ws-02.wade0426.me/v1", model="gemma-3-27b-it", **common_kwargs)

# 2. 定義鏈
chain = RunnableParallel({
    "linkedin": ChatPromptTemplate.from_template("寫一段關於{topic}的職場貼文") | llm_model | StrOutputParser(),
    "instagram": ChatPromptTemplate.from_template("寫一段關於{topic}的IG貼文") | vlm_model | StrOutputParser()
})

async def main():
    topic = input("輸入主題：")
    
    # 第一部分：流式輸出:需看到不同主題交錯
    print("\n[流式輸出 (需看到不同主題交錯)]")
    
    # 使用兩個列表來存儲暫存的 token
    cache = {"linkedin": [], "instagram": []}
    
    # 直接迭代 astream，並即時印出以確保「交錯感」
    async for chunk in chain.astream({"topic": topic}):
        for key, value in chunk.items():
            if value:
                # 每個 chunk 只輸出一次，並立即換行以達成垂直交錯格式
                print(f"{{'{key}': '{value.strip()}'}}")
                # 增加微小延遲，避免 Linux 執行太快導致視覺上沒交錯
                await asyncio.sleep(0.02)

    # 第二部分：批次處理:紀錄處理時間 
    print("\n" + "="*50)
    print("批次處理")
    
    start_time = time.time()  # 開始計時
    
    # 使用 ainvoke 同時執行所有並行任務
    result = await chain.ainvoke({"topic": topic})
    
    end_time = time.time()    # 結束計時
    
    print(f"耗時：{end_time - start_time:.2f} 秒")
    print("-" * 50)
    print(f"【LinkedIn 專家說】：\n{result['linkedin']}\n")
    print(f"【IG 網紅說】：\n{result['instagram']}\n")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())