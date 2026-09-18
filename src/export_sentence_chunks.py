import json
from pathlib import Path

from src.inspect_sentence_split import (
    split_sentences,
    group_sentence_records,
)


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"

    input_path = processed / "attachment_pages_sections.json"
    output_path = processed / "attachment_chunks_sentence_v2.json"

    pages = json.loads(input_path.read_text(encoding="utf-8"))
    all_chunks = []

    for page in pages:
        for section in page["sections"]:
            if not section["searchable"]:
                continue

            sentences = split_sentences(section["text"])

            if " ".join(sentences) != " ".join(section["text"].split()):
                raise ValueError(
                    f"{section['section_id']}: 문장 분리 중 본문이 바뀌었습니다."
                )

            # 이 함수 안에서 겹침을 제외한 문장 복원을 검사한다.
            records = group_sentence_records(
                sentences,
                target_words=200,
                overlap=True,
            )

            for number, record in enumerate(records, start=1):
                start = record["sentence_start"]
                overlap_count = record["overlap_sentences"]

                overlap_words = sum(
                    len(sentence.split())
                    for sentence in sentences[start:start + overlap_count]
                )

                chunk = {
                    "chunk_id": (
                        f"{section['section_id']}_sentencev2_c{number}"
                    ),
                    "document_id": page["document_id"],
                    "source_file": page["source_file"],
                    "pdf_page": page["pdf_page"],
                    "section_id": section["section_id"],
                    "section_kind": section["kind"],
                    "cleaning_version": page["cleaning_version"],
                    "chunking_version": "sentence_v2",
                    "chunk_index": number,
                    "target_words": 200,
                    "overlap_words": overlap_words,
                    **record,
                }

                if chunk["word_count"] != len(chunk["text"].split()):
                    raise ValueError("저장할 단어 수가 본문과 다릅니다.")

                all_chunks.append(chunk)

            print(
                f"PDF {page['pdf_page']}쪽 | {section['kind']} | "
                f"{len(records)}개 청크 | 문장 복원 통과"
            )

    if not all_chunks:
        raise ValueError("생성된 청크가 없습니다.")

    ids = [chunk["chunk_id"] for chunk in all_chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("청크 ID가 중복됩니다.")

    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    if saved != all_chunks:
        raise ValueError("저장한 파일이 생성 결과와 다릅니다.")

    print("\n전체 청크 수:", len(all_chunks))
    print("청크 ID 중복 및 저장 검증 통과")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()