import json
from pathlib import Path

def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap은 0 이상, chunk_size 미만이어야 합니다.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        # TODO 1: words에서 start부터 chunk_size개 단어를 선택
        chunk_words = words[start:start + chunk_size]

        chunks.append(" ".join(chunk_words))

        # 마지막 단어까지 포함했다면 종료
        if start + chunk_size >= len(words):
            break

        # TODO 2: 다음 조각의 시작 위치로 이동
        start = start + chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample = "A B C D E F G H I J K L"
    chunks = chunk_text(sample, chunk_size=5, overlap=2)

    for number, chunk in enumerate(chunks, start=1):
        print(number, chunk)

    assert chunks == [
        "A B C D E",
        "D E F G H",
        "G H I J K",
        "J K L",
    ]

    print("청킹 및 겹침 확인 완료")


    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    input_path = processed_dir / "attachment_pages_body.json"
    output_path = processed_dir / "attachment_chunks.json"

    pages = json.loads(input_path.read_text(encoding="utf-8"))
    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["body_text"],
            chunk_size=200,
            overlap=40,
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunk_data = {
                "chunk_id": (
                    f"{page['document_id']}"
                    f"_p{page['pdf_page']}"
                    f"_c{chunk_number}"
                ),
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "pdf_page": page["pdf_page"],
                "chunk_index": chunk_number,
                "chunk_size_words": 200,
                "overlap_words": 40,
                "word_count": len(chunk.split()),
                "text": chunk,
            }

            all_chunks.append(chunk_data)

        print(
            f"PDF {page['pdf_page']}쪽: "
            f"{len(page_chunks)}개 청크"
        )

    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("전체 청크 수:", len(all_chunks))
    print("저장 위치:", output_path)



    saved_chunks = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert len(saved_chunks) == len(all_chunks)

    chunk_ids = [chunk["chunk_id"] for chunk in saved_chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "청크 ID가 중복됩니다."

    for page in pages:
        page_chunks = [
            chunk
            for chunk in saved_chunks
            if chunk["document_id"] == page["document_id"]
            and chunk["pdf_page"] == page["pdf_page"]
        ]

        page_chunks.sort(key=lambda chunk: chunk["chunk_index"])
        original_words = page["body_text"].split()

        if not original_words:
            assert not page_chunks
            continue

        assert page_chunks, "본문이 있는데 청크가 없습니다."

        reconstructed_words = []

        for index, chunk in enumerate(page_chunks):
            words = chunk["text"].split()

            assert 0 < len(words) <= 200, "청크 크기가 범위를 벗어납니다."
            assert chunk["word_count"] == len(words)
            assert chunk["source_file"] == page["source_file"]

            if index == 0:
                reconstructed_words.extend(words)
            else:
                previous_words = page_chunks[index - 1]["text"].split()

                assert previous_words[-40:] == words[:40], (
                    "인접 청크의 40단어 겹침이 일치하지 않습니다."
                )

                reconstructed_words.extend(words[40:])

        assert reconstructed_words == original_words, (
            "청크를 합친 결과가 페이지의 원래 단어 순서와 다릅니다."
        )

        print(f"PDF {page['pdf_page']}쪽: 크기·겹침·본문 복원 통과")

    print("전체 청크 검증 완료")