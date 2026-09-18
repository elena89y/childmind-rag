import json
from pathlib import Path


# 새 청크 본문을 확인해 연결한 개발용 근거 목록
LABELS = {
    "q01": ["p2_s1_cleanv1_c1", "p2_s1_cleanv1_c2"],
    "q02": ["p2_s1_cleanv1_c2", "p2_s1_cleanv1_c3"],
    "q03": ["p3_s1_cleanv1_c1"],
    "q04": ["p5_s1_cleanv1_c2"],
    "q05": [
        "p1_s1_cleanv1_c1",
        "p1_s1_cleanv1_c2",
        "p5_s1_cleanv1_c1",
        "p5_s1_cleanv1_c2",
    ],
    "q06": ["p2_s1_cleanv1_c1"],
    "q07": ["p2_s1_cleanv1_c3"],
    "q08": ["p2_s1_cleanv1_c3"],
    "q09": ["p2_s1_cleanv1_c3"],
    "q10": ["p2_s1_cleanv1_c4"],
    "q11": ["p1_s3_cleanv1_c1"],
    "q16": ["p4_s1_cleanv1_c3"],
    "q17": ["p4_s1_cleanv1_c4"],
    "q18": ["p5_s1_cleanv1_c1"],
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    evaluation_dir = project_root / "evaluation"
    processed_dir = project_root / "data" / "processed"

    questions = json.loads(
        (evaluation_dir / "questions_v2.json").read_text(
            encoding="utf-8"
        )
    )
    chunks = json.loads(
        (processed_dir / "attachment_chunks_sections.json").read_text(
            encoding="utf-8"
        )
    )

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    if len(chunks_by_id) != len(chunks):
        raise ValueError("새 청크 ID가 중복됩니다.")

    dev_questions = [
        question
        for question in questions
        if question.get("split") == "dev"
    ]

    question_ids = [q["question_id"] for q in dev_questions]

    if (
        len(question_ids) != len(set(question_ids))
        or set(question_ids) != set(LABELS)
    ):
        raise ValueError("개발용 질문 목록과 라벨 연결 목록이 다릅니다.")

    results = []

    for question in dev_questions:
        question_id = question["question_id"]
        document_id = question["document_id"]

        relevant_ids = [
            f"{document_id}_{suffix}"
            for suffix in LABELS[question_id]
        ]

        for chunk_id in relevant_ids:
            if chunk_id not in chunks_by_id:
                raise ValueError(f"존재하지 않는 청크: {chunk_id}")

            chunk = chunks_by_id[chunk_id]

            if chunk["document_id"] != document_id:
                raise ValueError("질문과 근거의 문서가 다릅니다.")

            if chunk["pdf_page"] not in question["evidence_pdf_pages"]:
                raise ValueError("근거 페이지와 새 청크 페이지가 다릅니다.")

        result = question.copy()
        result["original_relevant_chunk_ids"] = question[
            "relevant_chunk_ids"
        ].copy()
        result["relevant_chunk_ids"] = relevant_ids
        result["corpus_version"] = "sections_v1"
        result["label_status"] = "draft"
        results.append(result)

        print(f"{question_id}: 근거 {len(relevant_ids)}개 연결")

    output_path = evaluation_dir / "questions_dev_sections.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n개발용 질문 수:", len(results))
    print("청크 ID·문서·근거 페이지 검증 완료")
    print("기존 질문 파일과 최종 비교용 질문은 유지했습니다.")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()