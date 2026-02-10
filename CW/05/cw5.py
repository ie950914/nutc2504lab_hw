import pdfplumber
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown
import os

# 設定檔案路徑
PDF_FILE = "example.pdf"

def run_pdfplumber():
    print("正在執行 pdfplumber 轉換...")
    output_path = "output_plumber.md"
    with pdfplumber.open(PDF_FILE) as pdf:
        full_text = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"## Page {i+1}\n\n{text}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
    print(f"完成！存檔至: {output_path}")

def run_docling():
    print("正在執行 Docling 轉換...")
    output_path = "output_docling.md"
    converter = DocumentConverter()
    result = converter.convert(PDF_FILE)
    md_output = result.document.export_to_markdown()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_output)
    print(f"完成！存檔至: {output_path}")

def run_markitdown():
    print("正在執行 Markitdown 轉換...")
    output_path = "output_markitdown.md"
    md = MarkItDown()
    result = md.convert(PDF_FILE)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"完成！存檔至: {output_path}")

if __name__ == "__main__":
    if not os.path.exists(PDF_FILE):
        print(f"找不到檔案: {PDF_FILE}，請確認檔案是否存在。")
    else:
        run_pdfplumber()
        print("-" * 30)
        run_docling()
        print("-" * 30)
        run_markitdown()
        print("\n所有轉換作業已完成！")