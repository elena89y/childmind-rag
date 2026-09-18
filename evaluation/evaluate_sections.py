import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from evaluation.evaluate_hybrid import reciprocal_rank_fusion
from src.bm25_retriever import tokenize
from src.dense_retriever import DenseRetriever


def calculate_metrics(ranked_ids, relevant_ids, k=3):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            return {
                "hit": 1,
                "reciprocal_rank": 1 / rank,
                "first_relevant_rank": rank,
            }

    return {
        "hit": 0,
        "reciprocal_rank": 0.0,
        "first_relevant_rank": None,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    chunks_path = (
        project_root / "data" / "processed" / "attachment_chunks_sections.json"
    )
    questions_path = (
        project_root / "evaluation" / "questions_dev_sections.json"
    )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # 최종 비교용 질문은 이번 평가에서 제외한다.
    questions = [
        question
        for question in all_questions
        if question.get("split") == "dev"
    ]

    if not chunks or not questions:
        raise ValueError("청크 또는 개발용 질문이 없습니다.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    known_ids = set(chunk_ids)

    if len(known_ids) != len(chunk_ids):
        raise ValueError("중복된 청크 ID가 있습니다.")

    question_ids = [question["question_id"] for question in questions]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError("중복된 질문 ID가 있습니다.")

    for question in questions:
        relevant = question["relevant_chunk_ids"]

        if not relevant:
            raise ValueError(
                f"{question['question_id']}: 정답 청크가 없습니다."
            )

        unknown = set(relevant) - known_ids

        if unknown:
            raise ValueError(
                f"{question['question_id']}: 없는 청크 ID {sorted(unknown)}"
            )

        if not tokenize(question["question"]):
            raise ValueError(
                f"{question['question_id']}: 검색할 영어 단어가 없습니다."
            )

    print("검색 대상 청크 수:", len(chunks))
    print("개발용 질문 수:", len(questions))
    print("최종 비교용 질문은 실행하지 않습니다.")

    # 검색기는 한 번만 준비한다.
    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])
    dense = DenseRetriever(chunks)

    method_names = ("bm25", "dense", "hybrid")
    results = []

    for question in questions:
        query = question["question"]
        relevant_ids = question["relevant_chunk_ids"]

        # BM25 전체 순위
        scores = bm25.get_scores(tokenize(query))
        indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), index),
        )
        bm25_ids = [chunk_ids[index] for index in indices]

        # Dense 전체 순위
        dense_results = dense.retrieve(query, k=len(chunks))
        dense_ids = [
            item["chunk_id"]
            for item in dense_results
        ]

        # 두 전체 순위를 RRF로 결합
        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_constant=60,
        )
        hybrid_ids = [item["chunk_id"] for item in fused]

        rankings = {
            "bm25": bm25_ids,
            "dense": dense_ids,
            "hybrid": hybrid_ids,
        }

        row = {
            "question_id": question["question_id"],
            "question": query,
            "relevant_chunk_ids": relevant_ids,
            "label_status": question.get("label_status", "draft"),
            "methods": {},
        }

        print(f"\n=== {question['question_id']} ===")
        print("질문:", query)

        for method in method_names:
            ranked_ids = rankings[method]
            metrics = calculate_metrics(
                ranked_ids,
                relevant_ids,
                k=3,
            )

            row["methods"][method] = {
                **metrics,
                "top3_ids": ranked_ids[:3],
                "ranked_ids": ranked_ids,
            }

            print(
                f"{method}: "
                f"Hit@3={metrics['hit']} | "
                f"RR@3={metrics['reciprocal_rank']:.4f}"
            )

            for rank, chunk_id in enumerate(ranked_ids[:3], start=1):
                label = (
                    "관련 라벨 있음"
                    if chunk_id in relevant_ids
                    else "관련 라벨 없음"
                )
                print(f"  {rank}위 | {chunk_id} | {label}")

        results.append(row)

    summary = {}

    print("\n=== 개발용 질문 비교 결과 ===")

    for method in method_names:
        hit_at_3 = sum(
            row["methods"][method]["hit"]
            for row in results
        ) / len(results)

        mrr_at_3 = sum(
            row["methods"][method]["reciprocal_rank"]
            for row in results
        ) / len(results)

        summary[method] = {
            "hit_at_3": hit_at_3,
            "mrr_at_3": mrr_at_3,
        }

        print(
            f"{method}: "
            f"Hit@3={hit_at_3:.4f}, "
            f"MRR@3={mrr_at_3:.4f}"
        )

    output = {
        "dataset": "evaluation/questions_dev_sections.json",
        "split": "dev",
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "k": 3,
        "rrf_constant": 60,
        "note": "현재 draft 라벨 기준 개발용 평가",
        "summary": summary,
        "results": results,
    }

    output_path = (
        project_root
        / "data"
        / "processed"
        / "retrieval_dev_sections.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n현재 draft 라벨 기준이며 답변 정확도 평가는 아닙니다.")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()