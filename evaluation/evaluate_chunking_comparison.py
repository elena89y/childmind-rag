import gc
import hashlib
import json
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from evaluation.evaluate_hybrid import reciprocal_rank_fusion
from evaluation.evaluate_sections import calculate_metrics
from src.bm25_retriever import tokenize
from src.dense_retriever import DenseRetriever


CONFIGS = [
    ("original", "attachment_chunks.json"),
    ("sections", "attachment_chunks_sections.json"),
    ("sentence_v2", "attachment_chunks_sentence_v2.json"),
]

METHODS = ("bm25", "dense", "hybrid")
K = 3
RRF_CONSTANT = 60


def load_input(path):
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    return data, digest


def validate(chunks, questions, version):
    if not chunks or len(questions) != 14:
        raise ValueError(
            f"{version}: 청크 또는 개발용 질문 수를 확인해주세요."
        )

    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    if len(by_id) != len(chunks):
        raise ValueError(f"{version}: 청크 ID가 중복됩니다.")

    question_ids = [q["question_id"] for q in questions]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError(f"{version}: 질문 ID가 중복됩니다.")

    for question in questions:
        qid = question["question_id"]

        if question.get("split") != "dev":
            raise ValueError(f"{qid}: 개발용 질문이 아닙니다.")

        if question.get("label_version") != "comparison_v1":
            raise ValueError(f"{qid}: 비교용 라벨 버전이 다릅니다.")

        if question.get("corpus_version") != version:
            raise ValueError(f"{qid}: 코퍼스 버전이 다릅니다.")

        if not tokenize(question["question"]):
            raise ValueError(f"{qid}: 검색할 영어 단어가 없습니다.")

        relevant = question["relevant_chunk_ids"]

        if not relevant or len(relevant) != len(set(relevant)):
            raise ValueError(f"{qid}: 정답 라벨이 비었거나 중복됩니다.")

        for chunk_id in relevant:
            if chunk_id not in by_id:
                raise ValueError(f"{qid}: 없는 청크 {chunk_id}")

            chunk = by_id[chunk_id]

            if chunk["document_id"] != question["document_id"]:
                raise ValueError(f"{qid}: 근거 문서가 다릅니다.")

            if chunk["pdf_page"] not in question["evidence_pdf_pages"]:
                raise ValueError(f"{qid}: 근거 페이지가 다릅니다.")


def evaluate(chunks, questions):
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])
    dense = DenseRetriever(chunks)

    results = []

    for question in questions:
        query = question["question"]
        relevant = question["relevant_chunk_ids"]

        scores = bm25.get_scores(tokenize(query))
        indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), index),
        )
        bm25_ids = [chunk_ids[index] for index in indices]

        dense_ids = [
            item["chunk_id"]
            for item in dense.retrieve(query, k=len(chunks))
        ]

        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_constant=RRF_CONSTANT,
        )

        rankings = {
            "bm25": bm25_ids,
            "dense": dense_ids,
            "hybrid": [item["chunk_id"] for item in fused],
        }

        row = {
            "question_id": question["question_id"],
            "question": query,
            "relevant_chunk_ids": relevant,
            "label_status": question["label_status"],
            "methods": {},
        }

        for method, ranked_ids in rankings.items():
            if (
                len(ranked_ids) != len(chunk_ids)
                or set(ranked_ids) != set(chunk_ids)
            ):
                raise ValueError(f"{method}: 전체 검색 순위가 잘못됐습니다.")

            metrics = calculate_metrics(ranked_ids, relevant, k=K)

            row["methods"][method] = {
                **metrics,
                "top3_ids": ranked_ids[:K],
                "ranked_ids": ranked_ids,
            }

        results.append(row)

        hybrid = row["methods"]["hybrid"]
        print(
            f"{question['question_id']} | Hybrid "
            f"Hit@3={hybrid['hit']} | "
            f"RR@3={hybrid['reciprocal_rank']:.4f}",
            flush=True,
        )

    summary = {}

    for method in METHODS:
        summary[method] = {
            "hit_at_3": sum(
                row["methods"][method]["hit"]
                for row in results
            ) / len(results),
            "mrr_at_3": sum(
                row["methods"][method]["reciprocal_rank"]
                for row in results
            ) / len(results),
        }

    return results, summary


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"
    prepared = []
    reference_questions = None

    # 모델을 불러오기 전에 세 버전의 입력부터 모두 검사한다.
    for version, corpus_name in CONFIGS:
        corpus_path = processed / corpus_name
        questions_path = (
            root / "evaluation"
            / f"questions_dev_{version}_comparison.json"
        )

        chunks, corpus_hash = load_input(corpus_path)
        questions, questions_hash = load_input(questions_path)
        validate(chunks, questions, version)

        signature = {
            q["question_id"]: (
                q["question"],
                q["document_id"],
                q["label_policy"],
            )
            for q in questions
        }

        if reference_questions is None:
            reference_questions = signature
        elif signature != reference_questions:
            raise ValueError("버전 사이의 질문 또는 라벨 정책이 다릅니다.")

        prepared.append({
            "version": version,
            "chunks": chunks,
            "questions": sorted(
                questions,
                key=lambda q: q["question_id"],
            ),
            "metadata": {
                "corpus": str(corpus_path.relative_to(root)),
                "corpus_sha256": corpus_hash,
                "dataset": str(questions_path.relative_to(root)),
                "dataset_sha256": questions_hash,
            },
        })

    output_dir = (
        processed / "chunking_comparisons" / str(time.time_ns())
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    comparison = {}

    print("개발용 14개 질문으로 세 버전을 비교합니다.")
    print("최종 비교용 질문과 답변 생성은 실행하지 않습니다.")
    print("저장 폴더:", output_dir, flush=True)

    for item in prepared:
        version = item["version"]

        print(
            f"\n=== {version} | "
            f"{len(item['chunks'])}개 청크 ===",
            flush=True,
        )

        results, summary = evaluate(
            item["chunks"],
            item["questions"],
        )

        output = {
            **item["metadata"],
            "corpus_version": version,
            "label_version": "comparison_v1",
            "split": "dev",
            "question_count": len(item["questions"]),
            "chunk_count": len(item["chunks"]),
            "k": K,
            "rrf_constant": RRF_CONSTANT,
            "note": "재검토한 초안 라벨 기준이며 답변 정확도 평가는 아님",
            "summary": summary,
            "results": results,
        }

        (output_dir / f"{version}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        comparison[version] = summary

        # 이전 버전의 검색기 객체 정리를 돕는다.
        gc.collect()

    print("\n=== 같은 라벨 기준의 검색 비교 ===")
    print("버전 | 검색 방식 | Hit@3 | MRR@3")

    for version, summary in comparison.items():
        for method in METHODS:
            metrics = summary[method]
            print(
                f"{version} | {method} | "
                f"{metrics['hit_at_3']:.4f} | "
                f"{metrics['mrr_at_3']:.4f}"
            )

    (output_dir / "summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n이전 평가와 라벨 버전이 다르므로 점수를 구분해서 기록하세요.")
    print("저장 위치:", output_dir)


if __name__ == "__main__":
    main()