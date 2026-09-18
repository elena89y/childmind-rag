import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in normalized.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    sample = "  hello    world  \nsecond\tline\n\nthird line  "
    result = clean_text(sample)

    print("정제 전:", repr(sample))
    print("정제 후:", repr(result))

    assert result == "hello world\nsecond line\n\nthird line"
    print("공백 정리 및 줄바꿈 보존 확인")

    project_root = Path(__file__).resolve().parent.parent
    input_path = (
        project_root / "data" / "processed" / "attachment_pages.json"
    )
    output_path = (
        project_root / "data" / "processed" / "attachment_pages_cleaned.json"
    )

    pages = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned_pages = []

    for page in pages:
        raw_text = page["text"]
        cleaned_text = clean_text(raw_text)

        cleaned_page = {
            "document_id": page["document_id"],
            "source_file": page["source_file"],
            "pdf_page": page["pdf_page"],
            "raw_text": raw_text,
            "clean_text": cleaned_text,
        }

        cleaned_pages.append(cleaned_page)

        print(
            f"PDF {page['pdf_page']}쪽: "
            f"{len(raw_text)} → {len(cleaned_text)}글자"
        )

    output_path.write_text(
        json.dumps(cleaned_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("저장 위치:", output_path)



    saved_pages = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(saved_pages) == len(pages), "페이지 수가 달라졌습니다."

    for original, saved in zip(pages, saved_pages):
        for key in ("document_id", "source_file", "pdf_page"):
            assert original[key] == saved[key], f"{key} 정보가 달라졌습니다."

        assert original["text"] == saved["raw_text"], "원문이 변경됐습니다."

        original_text = original["text"]
        cleaned_text = saved["clean_text"]

        assert (
            re.sub(r"\s+", "", original_text)
            == re.sub(r"\s+", "", cleaned_text)
        ), "공백 외의 문자가 변경됐습니다."

        assert (
            original_text.count("\n") == cleaned_text.count("\n")
        ), "줄바꿈 수가 달라졌습니다."

    print("검증 완료: 페이지 정보·원문·공백 외 문자·줄바꿈 수 보존")