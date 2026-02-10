import os
import logging
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# 定義檔案路徑
source_pdf = "sample_table.pdf"

# 定義 olmocr2_vlm_options (參考範例程式碼)
def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    hostname_and_port: str = "ws-01.wade0426.me", # 移除 https:// 與 /v1/ 以符合範例邏輯
    prompt: str = "Convert this page to clean, readable markdown format.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",
) -> ApiVlmOptions:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
   
    options = ApiVlmOptions(
        url=f"https://{hostname_and_port}/v1/chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=120,   
        scale=2.0,   
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

# --- 任務 4: 關閉 OCR (純文字提取) ---
def run_task_4():
    print("--- 執行任務 4: Docling (OCR Off) ---")
    pdf_options = PdfPipelineOptions(do_ocr=False) # 範例要求關閉 OCR
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )
    
    result = doc_converter.convert(source_pdf)
    with open("output_task4.md", "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
    print("任務 4 完成 (結果可能因關閉 OCR 而內容較少)")

# --- 任務 5: 使用 olmOCR-2 (VLM Pipeline) ---
def run_task_5():
    print("--- 執行任務 5: Docling (olmOCR-2) ---")
    
    # 配置 VLM pipeline 選項
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True   # 必須啟用以呼叫遠端 API
    )
   
    # 設定 olmocr2 的 VLM 選項
    pipeline_options.vlm_options = olmocr2_vlm_options(
        hostname_and_port="ws-01.wade0426.me",
        api_key="" # 若有需要請填入
    )
   
    # 建立文件轉換器，注意這裡要指定 VlmPipeline
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline, # 指定使用 VLM 核心
            )
        }
    )
    
    result = doc_converter.convert(source_pdf)
    with open("output_task5.md", "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
    print("任務 5 完成，請檢查 output_task5.md")

if __name__ == "__main__":
    if os.path.exists(source_pdf):
        run_task_4()
        run_task_5()
    else:
        print(f"找不到檔案: {source_pdf}")