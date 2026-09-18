from pathlib import Path

from pypdf import PdfReader


project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "data" / "raw" / "attachment_temperament_2012.pdf"

reader = PdfReader(pdf_path)
print("전체 페이지 수:", len(reader.pages))

first_page = reader.pages[0]
text = first_page.extract_text() or ""

print("첫 페이지 글자 수:", len(text))
print(text[:1500])