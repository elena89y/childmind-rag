import json
from pathlib import Path

from src.chunker import chunk_text


CHUNK_SIZE = 200
OVERLAP = 40


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    input_path = processed_dir / "attachment_pages_sections.json"
    output_path = processed_dir / "attachment_chunks_sections.json"

    pages = json.loads(input_path.read_text(encoding="utf-8"))
    all_chunks = []

    for page in pages:
        page_chunk_count = 0

        for section in page["sections"]:
            if not section["searchable"]:
                continue

            texts = chunk_text(
                section["text"],
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
            )

            reconstructed_words = []
            previous_words = None

            for number, text in enumerate(texts, start=1):
                words = text.split()

                if not 0 < len(words) <= CHUNK_SIZE:
                    raise ValueError("청크 크기가 범위를 벗어났습니다.")

                if previous_words is None:
                    reconstructed_words.extend(words)
                else:
                    if previous_words[-OVERLAP:] != words[:OVERLAP]:
                        raise ValueError("청크의 겹침이 일치하지 않습니다.")

                    reconstructed_words.extend(words[OVERLAP:])

                previous_words = words

                all_chunks.append({
                    "chunk_id": (
                        f"{section['section_id']}_cleanv1_c{number}"
                    ),
                    "document_id": page["document_id"],
                    "source_file": page["source_file"],
                    "pdf_page": page["pdf_page"],
                    "section_id": section["section_id"],
                    "section_kind": section["kind"],
                    "cleaning_version": page["cleaning_version"],
                    "chunk_index": number,
                    "chunk_size_words": CHUNK_SIZE,
                    "overlap_words": OVERLAP,
                    "word_count": len(words),
                    "text": text,
                })

            if reconstructed_words != section["text"].split():
                raise ValueError(
                    f"{section['section_id']}: 영역 본문 복원 실패"
                )

            page_chunk_count += len(texts)

            print(
                f"PDF {page['pdf_page']}쪽 | "
                f"{section['kind']} | "
                f"{len(texts)}개 청크 | 크기·겹침·복원 통과"
            )

        if page_chunk_count == 0:
            print(
                f"PDF {page['pdf_page']}쪽: "
                "검색 유지 영역이 없어 청크를 생성하지 않았습니다."
            )

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("청크 ID가 중복됩니다.")

    if not all_chunks:
        raise ValueError("생성된 청크가 없습니다.")

    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

    if saved_chunks != all_chunks:
        raise ValueError("저장한 청크가 생성 결과와 다릅니다.")

    print("\n전체 청크 수:", len(all_chunks))
    print("청크 ID 중복 및 저장 검증 완료")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()