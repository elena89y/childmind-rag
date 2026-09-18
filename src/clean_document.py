import json
from pathlib import Path


TARGET_DOCUMENT = "attachment_temperament_2012"


def find_boundary(lines, prefix):
    matches = [
        index
        for index, line in enumerate(lines)
        if line.startswith(prefix)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"경계 문구를 정확히 한 번 찾아야 합니다: "
            f"{prefix!r}, 발견 횟수: {len(matches)}"
        )

    return matches[0]


def split_page(page):
    if page["document_id"] != TARGET_DOCUMENT:
        raise ValueError("이 규칙은 지정한 논문에만 적용할 수 있습니다.")

    lines = page["body_text"].split("\n")
    page_number = page["pdf_page"]

    if page_number == 1:
        author_start = find_boundary(
            lines,
            "Y oo Rha Hong, MD",
        )
        introduction_start = find_boundary(lines, "Introduction")

        if author_start >= introduction_start:
            raise ValueError("저자 정보와 서론의 순서가 예상과 다릅니다.")

        ranges = [
            ("abstract", True, 0, author_start),
            ("publication_metadata", False, author_start, introduction_start),
            ("body", True, introduction_start, len(lines)),
        ]

    elif page_number == 5:
        references_start = find_boundary(lines, "References")
        figure_start = find_boundary(lines, "• Spoilt, immature")
        caption_start = find_boundary(
            lines,
            "Fig. 2. Children’s characteristics",
        )

        if not references_start < figure_start < caption_start:
            raise ValueError("참고문헌과 그림의 순서가 예상과 다릅니다.")

        ranges = [
            ("body", True, 0, references_start),
            ("references", False, references_start, figure_start),
            ("figure", True, figure_start, len(lines)),
        ]

    elif page_number == 6:
        if not lines[0].startswith("Psychiatry 1997;36:637-44."):
            raise ValueError("6쪽 참고문헌 시작 부분이 예상과 다릅니다.")

        find_boundary(lines, "25. Fonagy")
        ranges = [
            ("references", False, 0, len(lines)),
        ]

    elif page_number in (2, 3, 4):
        ranges = [
            ("body", True, 0, len(lines)),
        ]

    else:
        raise ValueError(f"예상하지 못한 PDF 페이지: {page_number}")

    sections = []
    reconstructed_lines = []

    for number, (kind, searchable, start, end) in enumerate(ranges, start=1):
        selected_lines = lines[start:end]
        reconstructed_lines.extend(selected_lines)

        sections.append({
            "section_id": (
                f"{page['document_id']}_p{page_number}_s{number}"
            ),
            "kind": kind,
            "searchable": searchable,
            "start_line": start + 1,
            "end_line": end,
            "text": "\n".join(selected_lines),
        })

    # 검색 제외 영역까지 합치면 원문이 정확히 복원되어야 한다.
    if "\n".join(reconstructed_lines) != page["body_text"]:
        raise ValueError(f"PDF {page_number}쪽 원문 복원 실패")

    result = page.copy()
    result["cleaning_version"] = "sections_v1"
    result["sections"] = sections

    return result


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    input_path = processed_dir / "attachment_pages_body.json"
    output_path = processed_dir / "attachment_pages_sections.json"

    pages = json.loads(input_path.read_text(encoding="utf-8"))

    if sorted(page["pdf_page"] for page in pages) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("PDF 페이지 구성이 예상과 다릅니다.")

    results = [split_page(page) for page in pages]

    for page in results:
        print(f"\n=== PDF {page['pdf_page']}쪽 ===")

        for section in page["sections"]:
            status = "검색 유지" if section["searchable"] else "별도 보관"
            print(
                f"{section['kind']} | {status} | "
                f"{len(section['text'])}글자"
            )

        print("원문 복원 확인 완료")

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n기존 원문과 청크 파일은 유지했습니다.")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()