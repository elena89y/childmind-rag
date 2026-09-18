import json
from pathlib import Path

from evaluation.evaluate_bm25 import evaluate_ranking
from src.dense_retriever import DenseRetriever


if __name__ == "__main__":
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

    known_ids = {chunk["chunk_id"] for chunk in chunks}

    if len(known_ids) != len(chunks):
        raise ValueError("청크 ID가 중복됩니다.")

    for question in questions:
        relevant_ids = question["relevant_chunk_ids"]

        if not relevant_ids:
            raise ValueError(
                f"{question['question_id']}: 관련 청크 라벨이 비어 있습니다."
            )

        if set(relevant_ids) - known_ids:
            raise ValueError(
                f"{question['question_id']}: 존재하지 않는 청크 ID입니다."
            )

    retriever = DenseRetriever(chunks)
    k = 3
    results = []

    for question in questions:
        retrieved = retriever.retrieve(question["question"], k=k)
        retrieved_ids = [item["chunk_id"] for item in retrieved]

        metrics = evaluate_ranking(
            retrieved_ids,
            question["relevant_chunk_ids"],
            k,
        )

        ranked_results = []

        print(f"\n=== {question['question_id']} ===")
        print("질문:", question["question"])

        for rank, item in enumerate(retrieved, start=1):
            is_relevant = (
                item["chunk_id"] in question["relevant_chunk_ids"]
            )

            label = "관련 라벨 있음" if is_relevant else "관련 라벨 없음"

            print(
                f"{rank}위 | {item['chunk_id']} | "
                f"{item['score']:.4f} | {label}"
            )

            ranked_results.append({
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "is_labeled_relevant": is_relevant,
            })

        print(
            f"Hit@{k}: {metrics['hit']} | "
            f"RR@{k}: {metrics['reciprocal_rank']:.4f}"
        )

        results.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "label_status": question["label_status"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "retrieved": ranked_results,
            **metrics,
        })

    mean_hit = sum(item["hit"] for item in results) / len(results)
    mean_rr = (
        sum(item["reciprocal_rank"] for item in results) / len(results)
    )

    report = {
        "retriever": "Dense",
        "model": retriever.model_name,
        "normalize_embeddings": True,
        "similarity": "cosine",
        "k": k,
        "question_count": len(results),
        "evaluation_status": "provisional",
        "hit_at_k": mean_hit,
        "mrr_at_k": mean_rr,
        "results": results,
    }

    output_path = processed_dir / "dense_evaluation.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 전체 결과 ===")
    print(f"Hit@{k}: {mean_hit:.4f}")
    print(f"MRR@{k}: {mean_rr:.4f}")
    print("현재 draft 라벨 기준 잠정 평가")
    print("저장 위치:", output_path)