import hashlib
import json
import re
import time
from pathlib import Path

import src.rag as rag


RANKING_RUN = "1789734181188044400"

VERSIONS = {
    "sections": "영역 정제",
    "sentence_v2": "문장 단위",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"
    ranking_dir = processed / "chunking_comparisons" / RANKING_RUN

    datasets = {}
    input_hashes = {}
    reference_questions = None

    # 먼저 두 버전의 입력을 모두 검사한다.
    for version in VERSIONS:
        ranking_path = ranking_dir / f"{version}.json"
        evaluation = read_json(ranking_path)

        if (
            evaluation["split"] != "dev"
            or evaluation["k"] != 3
            or evaluation["label_version"] != "comparison_v1"
            or evaluation["corpus_version"] != version
        ):
            raise ValueError(f"{version}: 평가 설정이 다릅니다.")

        corpus_path = root / evaluation["corpus"]
        questions_path = root / evaluation["dataset"]

        for path, expected_hash in (
            (corpus_path, evaluation["corpus_sha256"]),
            (questions_path, evaluation["dataset_sha256"]),
        ):
            if file_hash(path) != expected_hash:
                raise ValueError(
                    f"검색 평가 이후 파일이 변경됐습니다: {path}"
                )

        chunks_list = read_json(corpus_path)
        question_list = read_json(questions_path)
        rows_list = evaluation["results"]

        chunks = {c["chunk_id"]: c for c in chunks_list}
        questions = {q["question_id"]: q for q in question_list}
        rows = {r["question_id"]: r for r in rows_list}

        if (
            len(chunks) != len(chunks_list)
            or len(questions) != len(question_list)
            or len(rows) != len(rows_list)
        ):
            raise ValueError(f"{version}: 중복된 ID가 있습니다.")

        if len(questions) != 14 or set(questions) != set(rows):
            raise ValueError(f"{version}: 개발용 질문 구성이 다릅니다.")

        for qid, question in questions.items():
            if question.get("split") != "dev":
                raise ValueError(f"{qid}: 개발용 질문이 아닙니다.")

            if question["question"] != rows[qid]["question"]:
                raise ValueError(f"{qid}: 질문 내용이 다릅니다.")

            top_ids = rows[qid]["methods"]["hybrid"]["top3_ids"]

            if len(top_ids) != 3 or len(set(top_ids)) != 3:
                raise ValueError(f"{qid}: 상위 3개 근거가 잘못됐습니다.")

            for chunk_id in top_ids:
                if chunk_id not in chunks:
                    raise ValueError(f"{qid}: 없는 청크 {chunk_id}")

        signature = {
            qid: (
                q["question"],
                q.get("expected_answer", ""),
            )
            for qid, q in questions.items()
        }

        if reference_questions is None:
            reference_questions = signature
        elif signature != reference_questions:
            raise ValueError("두 버전의 질문 또는 기대 답변이 다릅니다.")

        datasets[version] = {
            "questions": questions,
            "rows": rows,
            "chunks": chunks,
        }

        for path in (ranking_path, corpus_path, questions_path):
            input_hashes[str(path.relative_to(root))] = file_hash(path)

    run_dir = (
        processed / "chunking_answer_comparisons" / str(time.time_ns())
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # 프롬프트뿐 아니라 생성 옵션과 사용자 메시지 구성도 추적한다.
    rag_path = Path(rag.__file__).resolve()
    rag_bytes = rag_path.read_bytes()
    (run_dir / "rag_snapshot.py").write_bytes(rag_bytes)

    metadata = {
        "model": rag.MODEL_NAME,
        "system_prompt": rag.SYSTEM_PROMPT,
        "generation_code_sha256": hashlib.sha256(rag_bytes).hexdigest(),
        "ranking_run": RANKING_RUN,
        "split": "dev",
        "label_version": "comparison_v1",
        "question_count": 14,
        "planned_calls": 28,
        "retrieval": "저장된 Hybrid 상위 3개" if False else "저장된 Hybrid 상위 3개",
        "input_sha256": input_hashes,
        "note": "질문당 버전별 1회 생성이며 반복 실험은 아님",
    }
    save_json(run_dir / "metadata.json", metadata)

    report = [
        "# 영역 정제·문장 단위 답변 비교",
        "",
        "동일한 질문과 현재 생성 함수를 사용합니다.",
        "기대 답변은 검토용 초안이며 모델에는 전달하지 않습니다.",
        "질문당 한 번의 생성 결과이므로 변동 가능성이 있습니다.",
        "",
    ]

    report_path = run_dir / "review.md"
    question_ids = sorted(reference_questions)
    records = []
    completed = 0
    total = len(question_ids) * len(VERSIONS)

    print("전체 답변 생성 횟수:", total)
    print("저장 폴더:", run_dir, flush=True)

    for index, qid in enumerate(question_ids):
        question = datasets["sections"]["questions"][qid]

        report.extend([
            f"## {qid}",
            "",
            f"질문: {question['question']}",
            "",
            f"기대 답변 초안: {question.get('expected_answer', '')}",
            "",
        ])

        # 특정 버전이 매번 먼저 실행되지 않도록 순서를 번갈아 둔다.
        order = list(VERSIONS)
        if index % 2:
            order.reverse()

        for version in order:
            dataset = datasets[version]
            top_ids = dataset["rows"][qid]["methods"]["hybrid"]["top3_ids"]

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
                f"{qid} | {VERSIONS[version]} 생성 중",
                flush=True,
            )

            record = {
                "question_id": qid,
                "question": question["question"],
                "expected_answer_draft": question.get("expected_answer", ""),
                "version": version,
                "sources": sources,
            }

            started = time.perf_counter()

            try:
                response = rag.generate_answer(
                    question["question"],
                    sources,
                )
                record["ollama_response"] = response
                record["done_reason"] = response.get("done_reason")

                answer = response.get("message", {}).get("content", "").strip()

                if not answer:
                    raise ValueError("모델이 빈 답변을 반환했습니다.")

                cited = sorted({
                    int(number)
                    for number in re.findall(r"\[(\d+)\]", answer)
                })
                valid = {s["citation_number"] for s in sources}

                record.update({
                    "status": "ok",
                    "answer": answer,
                    "cited_numbers": cited,
                    "invalid_citation_numbers": sorted(set(cited) - valid),
                })

            except Exception as error:
                record.update({
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })

            record["generation_seconds"] = round(
                time.perf_counter() - started,
                3,
            )

            save_json(run_dir / f"{qid}_{version}.json", record)
            records.append(record)
            completed += 1

            report.extend([
                f"### {VERSIONS[version]}",
                "",
                record.get("answer", record.get("error", "")),
                "",
                f"처리 상태: {record['status']}",
                f"생성 종료 이유: {record.get('done_reason')}",
                f"없는 인용 번호: {record.get('invalid_citation_numbers', [])}",
                "",
                "#### 전달한 근거",
                "",
            ])

            for source in sources:
                report.extend([
                    f"**[{source['citation_number']}] "
                    f"{source['chunk_id']} | PDF {source['pdf_page']}쪽**",
                    "",
                    source["text"],
                    "",
                ])

            report.extend([
                "#### 사람 검토",
                "",
                "- 근거가 문맥상 완결돼 있는가:",
                "- 주체·부정·가능성 등 의미를 보존했는가:",
                "- 한국어 용어가 정확한가:",
                "- 인용 근거가 해당 주장을 뒷받침하는가:",
                "- 질문의 핵심에 충분히 답했는가:",
                "- 문제가 있는 답변 문장과 이유:",
                "",
            ])

            report_path.write_text(
                "\n".join(report),
                encoding="utf-8",
            )

    # 질문·근거·답변을 한 파일에도 모아 후속 평가에 사용한다.
    save_json(run_dir / "answers.json", records)

    errors = sum(r["status"] == "error" for r in records)
    length_limits = sum(r.get("done_reason") == "length" for r in records)

    print("\n답변 비교 자료 저장 완료")
    print("실패한 요청 수:", errors)
    print("길이 제한 종료 수:", length_limits)
    print("검토 파일:", report_path)
    print("형식 점검만으로 답변이나 인용의 정확성을 보장하지 않습니다.")


if __name__ == "__main__":
    main()