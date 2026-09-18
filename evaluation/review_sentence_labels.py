import json
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"

    questions = load_json(
        root / "evaluation" / "questions_dev_sections.json"
    )
    old_chunks = load_json(
        processed / "attachment_chunks_sections.json"
    )
    new_chunks = load_json(
        processed / "attachment_chunks_sentence_v2.json"
    )

    old_by_id = {chunk["chunk_id"]: chunk for chunk in old_chunks}

    lines = [
        "# 문장 단위 청크의 정답 근거 검토",
        "",
        "새 청크는 검토 후보이며 아직 정답 라벨이 아닙니다.",
        "기존 라벨도 초안이므로 질문을 뒷받침하는지 함께 확인합니다.",
        "",
    ]

    for question in questions:
        if question.get("split") != "dev":
            raise ValueError("개발용 질문만 검토할 수 있습니다.")

        qid = question["question_id"]
        relevant = [
            old_by_id[chunk_id]
            for chunk_id in question["relevant_chunk_ids"]
        ]

        # 같은 문서·영역의 새 청크를 검토 후보로 모은다.
        areas = {
            (chunk["document_id"], chunk["section_id"])
            for chunk in relevant
        }

        candidates = [
            chunk
            for chunk in new_chunks
            if (chunk["document_id"], chunk["section_id"]) in areas
        ]

        if not candidates:
            raise ValueError(f"{qid}: 새 청크 후보가 없습니다.")

        lines.extend([
            f"## {qid}",
            "",
            question["question"],
            "",
            "### 기존 정답 라벨의 본문",
            "",
        ])

        for chunk in relevant:
            lines.extend([
                f"#### {chunk['chunk_id']}",
                "",
                chunk["text"],
                "",
            ])

        lines.extend(["### 새 청크 후보", ""])

        for chunk in candidates:
            lines.extend([
                f"#### {chunk['chunk_id']}",
                "",
                f"PDF {chunk['pdf_page']}쪽",
                "",
                chunk["text"],
                "",
            ])

        lines.extend([
            "### 검토 기록",
            "",
            "- 선택할 새 청크 ID:",
            "- 질문을 직접 뒷받침하는 문장:",
            "- 기존 라벨과 판단이 달라졌다면 이유:",
            "",
        ])

        print(f"{qid}: 새 청크 후보 {len(candidates)}개")

    output_path = processed / "sentence_label_review.md"
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n검토 자료 저장 완료")
    print("정답 라벨 파일은 변경하지 않았습니다.")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()