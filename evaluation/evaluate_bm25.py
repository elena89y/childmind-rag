import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.bm25_retriever import tokenize


def evaluate_ranking(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict:
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
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


if __name__ == "__main__":
    # 평가 계산 확인: 정답이 2위인 경우와 없는 경우
    check = evaluate_ranking(["a", "b", "c"], ["b"], k=3)
    assert check["hit"] == 1
    assert check["reciprocal_rank"] == 0.5

    check = evaluate_ranking(["a", "b", "c"], ["d"], k=3)
    assert check["hit"] == 0
    assert check["reciprocal_rank"] == 0.0

    project_root = Path(__file__).resolve().parent.parent

    chunks_path = (
        project_root / "data" / "processed" / "attachment_chunks.json"
    )
    questions_path = project_root / "evaluation" / "questions.json"

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    if not chunks or not questions:
        raise ValueError("청크와 질문이 모두 필요합니다.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    known_ids = set(chunk_ids)

    if len(known_ids) != len(chunk_ids):
        raise ValueError("청크 ID가 중복됩니다.")

    # 정답 라벨이 비었거나 존재하지 않는 ID면 계산을 중단
    for question in questions:
        relevant_ids = question["relevant_chunk_ids"]

        if not relevant_ids:
            raise ValueError(
                f"{question['question_id']}: 관련 청크 라벨이 비어 있습니다."
            )

        unknown_ids = set(relevant_ids) - known_ids

        if unknown_ids:
            raise ValueError(
                f"{question['question_id']}: 없는 청크 ID {unknown_ids}"
            )

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    k = 3
    results = []

    for question in questions:
        # 기본 조건: what/is를 제외하지 않음
        query_tokens = tokenize(question["question"])

        if not query_tokens:
            raise ValueError(
                f"{question['question_id']}: 질문 토큰이 없습니다."
            )

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:k]

        retrieved_ids = [
            chunks[index]["chunk_id"]
            for index in ranked_indices
        ]

        metrics = evaluate_ranking(
            retrieved_ids,
            question["relevant_chunk_ids"],
            k,
        )

        retrieved = []

        print(f"\n=== {question['question_id']} ===")
        print("질문:", question["question"])

        for rank, index in enumerate(ranked_indices, start=1):
            chunk_id = chunks[index]["chunk_id"]
            is_relevant = chunk_id in question["relevant_chunk_ids"]

            retrieved.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "score": float(scores[index]),
                "is_labeled_relevant": is_relevant,
            })

            label = "관련 라벨 있음" if is_relevant else "관련 라벨 없음"

            print(
                f"{rank}위 | {chunk_id} | "
                f"{float(scores[index]):.4f} | {label}"
            )

        print(
            f"Hit@{k}: {metrics['hit']} | "
            f"RR@{k}: {metrics['reciprocal_rank']:.4f}"
        )

        results.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "label_status": question["label_status"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "retrieved": retrieved,
            **metrics,
        })

    mean_hit = sum(item["hit"] for item in results) / len(results)
    mean_rr = (
        sum(item["reciprocal_rank"] for item in results) / len(results)
    )

    report = {
        "retriever": "BM25Okapi",
        "query_stopwords_removed": False,
        "k": k,
        "question_count": len(results),
        "evaluation_status": "provisional",
        "hit_at_k": mean_hit,
        "mrr_at_k": mean_rr,
        "results": results,
    }

    output_path = (
        project_root / "data" / "processed" / "bm25_evaluation.json"
    )

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 전체 결과: 현재 라벨 기준 잠정 평가 ===")
    print("질문 수:", len(results))
    print(f"Hit@{k}: {mean_hit:.4f}")
    print(f"MRR@{k}: {mean_rr:.4f}")
    print("저장 위치:", output_path)