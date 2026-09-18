import hashlib
import json
import time
from pathlib import Path

from src.rag import MODEL_NAME, SYSTEM_PROMPT, generate_answer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed = project_root / "data" / "processed"

    configurations = {
        "original": (
            "retrieval_dev_baseline.json",
            "attachment_chunks.json",
        ),
        "cleaned": (
            "retrieval_dev_sections.json",
            "attachment_chunks_sections.json",
        ),
    }

    datasets = {}

    for version, (ranking_file, chunk_file) in configurations.items():
        evaluation = read_json(processed / ranking_file)

        if evaluation["split"] != "dev":
            raise ValueError("개발용 평가 결과만 사용할 수 있습니다.")

        datasets[version] = {
            "rows": {
                row["question_id"]: row
                for row in evaluation["results"]
            },
            "chunks": {
                chunk["chunk_id"]: chunk
                for chunk in read_json(processed / chunk_file)
            },
        }

    original_rows = datasets["original"]["rows"]
    cleaned_rows = datasets["cleaned"]["rows"]

    if not original_rows or set(original_rows) != set(cleaned_rows):
        raise ValueError("두 버전의 질문 목록이 다릅니다.")

    for question_id, row in original_rows.items():
        if row["question"] != cleaned_rows[question_id]["question"]:
            raise ValueError(f"{question_id}: 두 버전의 질문이 다릅니다.")

    questions = {
        item["question_id"]: item
        for item in read_json(
            project_root / "evaluation" / "questions_v2.json"
        )
        if item.get("split") == "dev"
    }

    if set(questions) != set(original_rows):
        raise ValueError("현재 개발용 질문과 저장된 검색 결과가 다릅니다.")

    for question_id, question in questions.items():
        if question["question"] != original_rows[question_id]["question"]:
            raise ValueError(f"{question_id}: 질문이 변경됐습니다.")

    # 실행마다 새 폴더에 저장해 이전 결과를 보존한다.
    run_dir = (
        processed / "answer_comparisons" / str(time.time_ns())
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "model": MODEL_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "split": "dev",
        "question_count": len(questions),
        "retrieval": "저장된 Hybrid 상위 3개",
        "input_sha256": {
            name: hashlib.sha256(
                (processed / name).read_bytes()
            ).hexdigest()
            for files in configurations.values()
            for name in files
        },
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 기존·정제 버전 답변 비교",
        "",
        "동일한 질문과 생성 프롬프트를 사용합니다.",
        "근거와 답변은 아래에 전체 저장합니다.",
        "정성 평가는 아직 작성하지 않은 상태입니다.",
        "",
    ]

    report_path = run_dir / "review.md"
    total = len(questions) * 2
    completed = 0

    print("전체 답변 생성 횟수:", total)
    print("저장 폴더:", run_dir, flush=True)

    for question_id, question in questions.items():
        report.extend([
            f"## {question_id}",
            "",
            f"질문: {question['question']}",
            "",
            f"기대 답변 초안: {question['expected_answer']}",
            "",
        ])

        for version, version_name in (
            ("original", "기존 버전"),
            ("cleaned", "정제 버전"),
        ):
            dataset = datasets[version]
            row = dataset["rows"][question_id]
            top_ids = row["methods"]["hybrid"]["top3_ids"]

            sources = []

            for number, chunk_id in enumerate(top_ids, start=1):
                chunk = dataset["chunks"][chunk_id]
                sources.append({
                    "citation_number": number,
                    "chunk_id": chunk_id,
                    "document": chunk["source_file"],
                    "pdf_page": chunk["pdf_page"],
                    "text": chunk["text"],
                })

            print(
                f"[{completed + 1}/{total}] "
                f"{question_id} | {version_name} 생성 중",
                flush=True,
            )

            started = time.perf_counter()
            response = generate_answer(question["question"], sources)
            elapsed = time.perf_counter() - started

            answer = response.get("message", {}).get("content", "").strip()

            if not answer:
                raise ValueError(
                    f"{question_id} {version}: 빈 답변입니다."
                )

            record = {
                "question_id": question_id,
                "question": question["question"],
                "version": version,
                "answer": answer,
                "sources": sources,
                "done_reason": response.get("done_reason"),
                "generation_seconds": round(elapsed, 3),
                "ollama_response": response,
            }

            (run_dir / f"{question_id}_{version}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report.extend([
                f"### {version_name}",
                "",
                answer,
                "",
                f"생성 종료 이유: {response.get('done_reason')}",
                "",
                "#### 전달한 근거",
                "",
            ])

            for source in sources:
                report.extend([
                    f"**[{source['citation_number']}] "
                    f"{source['chunk_id']} | "
                    f"PDF {source['pdf_page']}쪽**",
                    "",
                    source["text"],
                    "",
                ])

            report.extend([
                "#### 사람 검토 — 미작성",
                "",
                "- 근거 완결성:",
                "- 답변이 근거의 의미를 보존하는가:",
                "- 각 인용이 해당 주장을 뒷받침하는가:",
                "- 질문의 핵심에 충분히 답했는가:",
                "- 출처 가독성:",
                "- 문제가 있는 답변 문장과 근거:",
                "",
            ])

            # 중간에 오류가 나도 완료된 결과는 남긴다.
            report_path.write_text(
                "\n".join(report),
                encoding="utf-8",
            )

            completed += 1

    print("\n답변 비교 자료 저장 완료")
    print("검토 파일:", report_path)
    print("아직 어느 버전이 더 좋은지 자동 판정하지 않았습니다.")


if __name__ == "__main__":
    main()