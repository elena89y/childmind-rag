import json
from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        page_data = {
            "document_id": pdf_path.stem,
            "source_file": pdf_path.name,
            "pdf_page": page_number,
            "text": text,
        }

        pages.append(page_data)

    return pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "attachment_temperament_2012.pdf"
    )

    pages = load_pdf(pdf_path)

    print("추출한 페이지 수:", len(pages))

    for item in pages:
        print(
            item["source_file"],
            item["pdf_page"],
            len(item["text"]),
        )


    output_path = (
        project_root / "data" / "processed" / "attachment_pages.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("저장 위치:", output_path)