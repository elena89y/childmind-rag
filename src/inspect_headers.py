import json
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
input_path = (
    project_root / "data" / "processed" / "attachment_pages_cleaned.json"
)

pages = json.loads(input_path.read_text(encoding="utf-8"))

for page in pages:
    print(f"\n=== PDF {page['pdf_page']}쪽: 처음 8줄 ===")

    lines = page["clean_text"].split("\n")

    for line_number, line in enumerate(lines[:8], start=1):
        print(f"{line_number}: {line!r}")