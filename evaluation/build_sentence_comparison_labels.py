import json
from pathlib import Path


# 새 청크 본문을 검토해 정한 개발용 라벨 초안
SENTENCE_LABELS = {
    "q01": [
        "p2_s1_sentencev2_c1",
        "p2_s1_sentencev2_c2",
        "p2_s1_sentencev2_c3",
    ],
    "q02": [
        "p2_s1_sentencev2_c2",
        "p2_s1_sentencev2_c3",
    ],
    "q03": ["p3_s1_sentencev2_c1"],
    "q04": ["p5_s1_sentencev2_c2"],
    "q05": [
        "p1_s1_sentencev2_c1",
        "p1_s1_sentencev2_c2",
        "p5_s1_sentencev2_c1",
        "p5_s1_sentencev2_c2",
        "p5_s1_sentencev2_c3",
    ],
    "q06": ["p2_s1_sentencev2_c1"],
    "q07": ["p2_s1_sentencev2_c3"],
    "q08": ["p2_s1_sentencev2_c3"],
    "q09": ["p2_s1_sentencev2_c3"],
    "q10": ["p2_s1_sentencev2_c4"],
    "q11": ["p1_s3_sentencev2_c1"],
    "q16": ["p4_s1_sentencev2_c3"],
    "q17": ["p4_s1_sentencev2_c4"],
    "q18": ["p5_s1_sentencev2_c1"],
}

POLICY = (
    "질문 답변의 일부를 직접 뒷받침하는 명확한 본문을 관련 근거로 인정한다. "
    "단순 주제 일치나 관계가 불명확한 표·그림 텍스트는 인정하지 않는다."
)

REVIEW_NOTES = {
    "q01": "안정 애착의 탐색·분리·재회 행동 설명을 관련 근거로 포함.",
    "q05": (
        "아이의 동기에 반응하는 양육 설명을 부분 근거로 포함. "
        "본문에 명시된 스트레스 등의 조건을 유지해야 함."
    ),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path(__file__).resolve().parent.parent
    evaluation = root / "evaluation"
    processed = root / "data" / "processed"

    original_questions = [
        question
        for question in load_json(evaluation / "questions_v2.json")
        if question.get("split") == "dev"
    ]

    sections_questions = load_json(
        evaluation / "questions_dev_sections.json"
    )

    configs = [
        (
            "original",
            original_questions,
            "attachment_chunks.json",
            {
                "q01": "p2_c3",
                "q05": "p5_c3",
            },
        ),
        (
            "sections",
            sections_questions,
            "attachment_chunks_sections.json",
            {
                "q01": "p2_s1_cleanv1_c3",
                "q05": "p5_s1_cleanv1_c3",
            },
        ),
        (
            "sentence_v2",
            original_questions,
            "attachment_chunks_sentence_v2.json",
            {},
        ),
    ]

    # 모든 설정의 검증이 끝난 뒤 저장한다.
    pending_outputs = []

    for version, questions, corpus_name, additions in configs:
        question_ids = [q["question_id"] for q in questions]

        if (
            len(question_ids) != len(set(question_ids))
            or set(question_ids) != set(SENTENCE_LABELS)
        ):
            raise ValueError(
                f"{version}: 개발용 질문 구성이 다릅니다."
            )

        chunks = load_json(processed / corpus_name)
        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        if len(chunks_by_id) != len(chunks):
            raise ValueError(
                f"{version}: 청크 ID가 중복됩니다."
            )

        results = []

        for question in questions:
            if question.get("split") != "dev":
                raise ValueError(
                    "개발용 질문만 처리할 수 있습니다."
                )

            qid = question["question_id"]
            document_id = question["document_id"]

            if version == "sentence_v2":
                relevant_ids = [
                    f"{document_id}_{suffix}"
                    for suffix in SENTENCE_LABELS[qid]
                ]
            else:
                relevant_ids = question[
                    "relevant_chunk_ids"
                ].copy()

                if qid in additions:
                    relevant_ids.append(
                        f"{document_id}_{additions[qid]}"
                    )

            relevant_ids = sorted(set(relevant_ids))

            if not relevant_ids:
                raise ValueError(
                    f"{qid}: 연결된 정답 근거가 없습니다."
                )

            for chunk_id in relevant_ids:
                if chunk_id not in chunks_by_id:
                    raise ValueError(
                        f"존재하지 않는 청크: {chunk_id}"
                    )

                chunk = chunks_by_id[chunk_id]

                if chunk["document_id"] != document_id:
                    raise ValueError(
                        f"{qid}: 근거 문서가 다릅니다."
                    )

                if (
                    chunk["pdf_page"]
                    not in question["evidence_pdf_pages"]
                ):
                    raise ValueError(
                        f"{qid}: 근거 페이지 확인이 필요합니다."
                    )

            result = question.copy()

            result["previous_relevant_chunk_ids"] = question[
                "relevant_chunk_ids"
            ].copy()

            result["relevant_chunk_ids"] = relevant_ids
            result["corpus_version"] = version
            result["label_version"] = "comparison_v1"
            result["label_status"] = "draft"
            result["label_policy"] = POLICY
            result["label_review_note"] = REVIEW_NOTES.get(
                qid,
                "검토한 근거 내용을 청크 범위에 연결.",
            )

            results.append(result)

        output_path = (
            evaluation
            / f"questions_dev_{version}_comparison.json"
        )

        pending_outputs.append((output_path, results))

    for output_path, results in pending_outputs:
        output_path.write_text(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        saved = load_json(output_path)

        if saved != results:
            raise ValueError(
                f"{output_path.name}: 저장 검증에 실패했습니다."
            )

        print(
            f"{output_path.name}: "
            f"{len(results)}개 질문 저장"
        )

    print("\n청크 ID·문서·페이지 및 저장 검증 완료")
    print("기존 라벨 파일과 최종 비교용 질문은 변경하지 않았습니다.")
    print(
        "검토 범위 내의 초안이며 "
        "전체 근거를 빠짐없이 찾았다는 뜻은 아닙니다."
    )


if __name__ == "__main__":
    main()