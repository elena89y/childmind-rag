import json
from pathlib import Path


TARGET_DOCUMENT = "attachment_temperament_2012"


def remove_header(page: dict) -> dict:
    if page["document_id"] != TARGET_DOCUMENT:
        raise ValueError("이 머리말 규칙은 지정한 논문에만 적용할 수 있습니다.")

    lines = page["clean_text"].split("\n")
    page_number = page["pdf_page"]

    # 첫 페이지는 5줄, 나머지는 2줄
    header_count = 5 if page_number == 1 else 2

    header_lines = lines[:header_count]
    body_lines = lines[header_count:]

    # 인쇄된 페이지 번호가 예상한 머리말 영역에 있는지 확인
    expected_printed_page = str(448 + page_number)

    if expected_printed_page not in header_lines:
        raise ValueError(
            f"PDF {page_number}쪽의 머리말이 예상과 다릅니다."
        )

    result = page.copy()
    result["removed_header_lines"] = header_lines
    result["body_text"] = "\n".join(body_lines)

    # 삭제 영역과 남긴 영역을 합치면 입력과 정확히 같아야 함
    reconstructed = "\n".join(
        result["removed_header_lines"]
        + result["body_text"].split("\n")
    )
    assert reconstructed == page["clean_text"]

    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    input_path = processed_dir / "attachment_pages_cleaned.json"
    output_path = processed_dir / "attachment_pages_body.json"

    pages = json.loads(input_path.read_text(encoding="utf-8"))
    results = []

    for page in pages:
        result = remove_header(page)
        results.append(result)

        print(f"\n=== PDF {result['pdf_page']}쪽 ===")
        print("제거한 줄:", result["removed_header_lines"])
        print("본문 시작:", repr(result["body_text"][:100]))

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n저장 위치:", output_path)