import json
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from evaluation.evaluate_dev import calculate_metrics


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_COUNT = 10
MAX_LENGTH = 512


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed = project_root / "data" / "processed"

    baseline = json.loads(
        (processed / "retrieval_dev_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    chunks = json.loads(
        (processed / "attachment_chunks.json").read_text(
            encoding="utf-8"
        )
    )
    questions = json.loads(
        (project_root / "evaluation" / "questions_v2.json").read_text(
            encoding="utf-8"
        )
    )

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }
    dev_questions = {
        question["question_id"]: question
        for question in questions
        if question.get("split") == "dev"
    }
    rows = baseline["results"]

    # 保存した基準結果ではなく現在の条件と一致するか確認する。
    if baseline.get("split") != "dev":
        raise ValueError("개발용 기준 결과가 아닙니다.")

    if not rows:
        raise ValueError("평가할 질문이 없습니다.")

    baseline_ids = [row["question_id"] for row in rows]

    if (
        len(set(baseline_ids)) != len(baseline_ids)
        or set(baseline_ids) != set(dev_questions)
    ):
        raise ValueError("기준 결과와 현재 개발용 질문 목록이 다릅니다.")

    for row in rows:
        current = dev_questions[row["question_id"]]

        if (
            row["question"] != current["question"]
            or set(row["relevant_chunk_ids"])
            != set(current["relevant_chunk_ids"])
        ):
            raise ValueError(
                "질문 또는 라벨이 변경됐습니다. "
                "evaluate_dev를 먼저 다시 실행해주세요."
            )

        ranked_ids = row["methods"]["hybrid"]["ranked_ids"]

        if (
            len(ranked_ids) != len(chunks_by_id)
            or set(ranked_ids) != set(chunks_by_id)
        ):
            raise ValueError("기준 결과와 현재 청크 목록이 다릅니다.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    print("개발용 질문 수:", len(rows))
    print("질문별 재정렬 후보 수:", CANDIDATE_COUNT)
    print("모델 불러오는 중:", MODEL_NAME, flush=True)

    model = CrossEncoder(
        MODEL_NAME,
        device="cuda",
        max_length=MAX_LENGTH,
    )

    # 첫 실행 준비 비용이 질문별 측정에 섞이지 않도록 예열
    model.predict(
        [("What is parenting?", "Parents care for their children.")],
        show_progress_bar=False,
    )
    torch.cuda.synchronize()

    results = []

    for row in rows:
        question = row["question"]
        relevant_ids = row["relevant_chunk_ids"]
        original_ids = row["methods"]["hybrid"]["ranked_ids"]
        candidate_ids = original_ids[:CANDIDATE_COUNT]

        pairs = [
            (question, chunks_by_id[chunk_id]["text"])
            for chunk_id in candidate_ids
        ]

        # 질문과 본문을 합친 토큰 수를 확인한다.
        encoded = model.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths = [
            len(token_ids)
            for token_ids in encoded["input_ids"]
        ]

        if max(lengths) > MAX_LENGTH:
            raise ValueError(
                f"{row['question_id']}: 질문과 청크의 입력 길이가 "
                f"{MAX_LENGTH}토큰을 초과했습니다."
            )

        torch.cuda.synchronize()
        started = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        # 동점이면 기존 Hybrid 순서를 유지한다.
        indices = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(scores[index]), index),
        )
        reranked_ids = [
            candidate_ids[index]
            for index in indices
        ]

        before = calculate_metrics(original_ids, relevant_ids, k=3)
        after = calculate_metrics(reranked_ids, relevant_ids, k=3)

        difference = (
            after["reciprocal_rank"] - before["reciprocal_rank"]
        )
        change = (
            "개선" if difference > 0
            else "악화" if difference < 0
            else "동일"
        )

        candidate_hit = int(
            bool(set(candidate_ids) & set(relevant_ids))
        )

        print(f"\n=== {row['question_id']} ===")
        print("질문:", question)
        print(
            f"RR@3: {before['reciprocal_rank']:.4f}"
            f" → {after['reciprocal_rank']:.4f} | {change}"
        )
        print("재정렬 소요 시간:", round(elapsed, 4), "초")

        for rank, index in enumerate(indices[:3], start=1):
            chunk_id = candidate_ids[index]
            label = (
                "관련 라벨 있음"
                if chunk_id in relevant_ids
                else "관련 라벨 없음"
            )
            print(
                f"{rank}위 | {chunk_id} | "
                f"{float(scores[index]):.4f} | {label}"
            )

        results.append({
            "question_id": row["question_id"],
            "question": question,
            "relevant_chunk_ids": relevant_ids,
            "candidate_ids": candidate_ids,
            "candidate_hit_at_10": candidate_hit,
            "before": before,
            "after": after,
            "change": change,
            "rerank_seconds": elapsed,
            "max_pair_tokens": max(lengths),
            "reranked": [
                {
                    "chunk_id": candidate_ids[index],
                    "score": float(scores[index]),
                }
                for index in indices
            ],
        })

    summary = {}

    print("\n=== 동일 조건 비교: 개발용 draft 라벨 ===")

    for key, name in (
        ("before", "hybrid"),
        ("after", "hybrid_reranker"),
    ):
        hit = sum(row[key]["hit"] for row in results) / len(results)
        mrr = sum(
            row[key]["reciprocal_rank"]
            for row in results
        ) / len(results)

        summary[name] = {
            "hit_at_3": hit,
            "mrr_at_3": mrr,
        }
        print(f"{name}: Hit@3={hit:.4f}, MRR@3={mrr:.4f}")

    for change in ("개선", "악화", "동일"):
        ids = [
            row["question_id"]
            for row in results
            if row["change"] == change
        ]
        print(f"{change}: {len(ids)}개 | {', '.join(ids) or '없음'}")

    mean_seconds = sum(
        row["rerank_seconds"]
        for row in results
    ) / len(results)

    print("평균 재정렬 소요 시간:", round(mean_seconds, 4), "초")
    print("모델 로딩·검색·답변 생성 시간은 포함하지 않습니다.")

    output_path = processed / "reranker_dev_evaluation.json"
    output_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "split": "dev",
                "candidate_count": CANDIDATE_COUNT,
                "max_length": MAX_LENGTH,
                "question_count": len(results),
                "summary": summary,
                "mean_rerank_seconds": mean_seconds,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()