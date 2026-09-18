import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from evaluation.evaluate_bm25 import evaluate_ranking
from src.bm25_retriever import tokenize
from src.dense_retriever import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_constant: int = 60,
) -> list[dict]:
    if rrf_constant <= 0:
        raise ValueError("RRF 상수는 0보다 커야 합니다.")

    fused_scores = {}

    for ranked_ids in ranked_lists:
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("검색 순위 목록에 중복 ID가 있습니다.")

        for rank, chunk_id in enumerate(ranked_ids, start=1):
            contribution = 1 / (rrf_constant + rank)

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + contribution
            )

    # 같은 점수일 때는 청크 ID 순서로 정렬
    ordered = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        {"chunk_id": chunk_id, "score": score}
        for chunk_id, score in ordered
    ]


if __name__ == "__main__":
    # 양쪽에서 1·2위가 뒤바뀐 두 문서는 같은 합산 점수를 받음
    sample = reciprocal_rank_fusion(
        [["a", "b"], ["b", "a"]],
        rrf_constant=60,
    )
    assert abs(sample[0]["score"] - sample[1]["score"]) < 1e-12

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    chunks = json.loads(
        (processed_dir / "attachment_chunks.json").read_text(
            encoding="utf-8"
        )
    )

    questions = json.loads(
        (project_root / "evaluation" / "questions.json").read_text(
            encoding="utf-8"
        )
    )

    if not chunks or not questions:
        raise ValueError("청크와 질문이 모두 필요합니다.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    known_ids = set(chunk_ids)

    if len(known_ids) != len(chunk_ids):
        raise ValueError("청크 ID가 중복됩니다.")

    for question in questions:
        relevant_ids = question["relevant_chunk_ids"]

        if not relevant_ids:
            raise ValueError(
                f"{question['question_id']}: 관련 라벨이 비어 있습니다."
            )

        if set(relevant_ids) - known_ids:
            raise ValueError(
                f"{question['question_id']}: 없는 청크 ID입니다."
            )

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    dense = DenseRetriever(chunks)

    evaluation_k = 3
    candidate_k = len(chunks)
    rrf_constant = 60

    results = []
    method_metrics = {
        "bm25": [],
        "dense": [],
        "hybrid": [],
    }

    for question in questions:
        query = question["question"]
        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("BM25 질문 토큰이 비어 있습니다.")

        # 1. BM25가 전체 청크에 매긴 순위
        bm25_scores = bm25.get_scores(query_tokens)

        bm25_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(bm25_scores[index]), index),
        )[:candidate_k]

        bm25_ids = [
            chunks[index]["chunk_id"]
            for index in bm25_indices
        ]

        # 2. Dense가 전체 청크에 매긴 순위
        dense_results = dense.retrieve(query, k=candidate_k)
        dense_ids = [
            item["chunk_id"]
            for item in dense_results
        ]

        # 3. 두 순위를 RRF로 결합
        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_constant=rrf_constant,
        )

        hybrid_ids = [
            item["chunk_id"]
            for item in fused
        ]

        rankings = {
            "bm25": bm25_ids,
            "dense": dense_ids,
            "hybrid": hybrid_ids,
        }

        question_metrics = {}

        for method, ranked_ids in rankings.items():
            metrics = evaluate_ranking(
                ranked_ids,
                question["relevant_chunk_ids"],
                evaluation_k,
            )
            method_metrics[method].append(metrics)
            question_metrics[method] = metrics

        bm25_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate(bm25_ids, start=1)
        }
        dense_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate(dense_ids, start=1)
        }

        print(f"\n=== {question['question_id']} ===")
        print("질문:", query)

        top_results = []

        for rank, item in enumerate(fused[:evaluation_k], start=1):
            chunk_id = item["chunk_id"]
            is_relevant = (
                chunk_id in question["relevant_chunk_ids"]
            )
            label = "관련 라벨 있음" if is_relevant else "관련 라벨 없음"

            print(
                f"{rank}위 | {chunk_id} | "
                f"RRF {item['score']:.6f} | "
                f"BM25 {bm25_ranks[chunk_id]}위 / "
                f"Dense {dense_ranks[chunk_id]}위 | "
                f"{label}"
            )

            top_results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "rrf_score": item["score"],
                "bm25_rank": bm25_ranks[chunk_id],
                "dense_rank": dense_ranks[chunk_id],
                "is_labeled_relevant": is_relevant,
            })

        hybrid_metrics = question_metrics["hybrid"]

        print(
            f"Hit@3: {hybrid_metrics['hit']} | "
            f"RR@3: {hybrid_metrics['reciprocal_rank']:.4f}"
        )

        results.append({
            "question_id": question["question_id"],
            "question": query,
            "label_status": question["label_status"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "rankings": rankings,
            "hybrid_top_results": top_results,
            "metrics": question_metrics,
        })

    summary = {}

    print("\n=== 동일 조건 비교: draft 라벨 기준 ===")

    for method, metrics_list in method_metrics.items():
        mean_hit = (
            sum(item["hit"] for item in metrics_list)
            / len(metrics_list)
        )
        mean_rr = (
            sum(item["reciprocal_rank"] for item in metrics_list)
            / len(metrics_list)
        )

        summary[method] = {
            "hit_at_k": mean_hit,
            "mrr_at_k": mean_rr,
        }

        print(
            f"{method}: "
            f"Hit@3={mean_hit:.4f}, MRR@3={mean_rr:.4f}"
        )

    report = {
        "evaluation_status": "provisional",
        "question_count": len(questions),
        "evaluation_k": evaluation_k,
        "candidate_k_per_retriever": candidate_k,
        "rrf_constant": rrf_constant,
        "dense_model": dense.model_name,
        "summary": summary,
        "results": results,
    }

    output_path = processed_dir / "hybrid_evaluation.json"

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("저장 위치:", output_path)