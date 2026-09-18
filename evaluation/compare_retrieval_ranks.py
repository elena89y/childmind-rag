import json
from pathlib import Path


def load_results(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if data["split"] != "dev" or data["k"] != 3:
        raise ValueError("개발용 상위 3개 평가 결과인지 확인해주세요.")

    rows = data["results"]
    indexed = {row["question_id"]: row for row in rows}

    if len(indexed) != len(rows):
        raise ValueError("질문 ID가 중복됩니다.")

    return indexed


def measure(row):
    relevant = set(row["relevant_chunk_ids"])
    ranked = row["methods"]["hybrid"]["ranked_ids"]

    rank = next(
        (
            number
            for number, chunk_id in enumerate(ranked, start=1)
            if chunk_id in relevant
        ),
        None,
    )

    # 상위 3개 밖에 있으면 RR@3은 0이다.
    rr = 1 / rank if rank is not None and rank <= 3 else 0.0
    return rank, rr


def show_top3(label, row):
    relevant = set(row["relevant_chunk_ids"])

    print(f"\n{label} 상위 3개")
    for rank, chunk_id in enumerate(
        row["methods"]["hybrid"]["ranked_ids"][:3],
        start=1,
    ):
        status = "관련 라벨 있음" if chunk_id in relevant else "관련 라벨 없음"
        print(f"  {rank}위 | {chunk_id} | {status}")

    print("정답 라벨:", ", ".join(sorted(relevant)))


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"

    old = load_results(processed / "retrieval_dev_baseline.json")
    new = load_results(processed / "retrieval_dev_sections.json")

    if not old or old.keys() != new.keys():
        raise ValueError("두 결과의 질문 ID 구성이 다르거나 비어 있습니다.")

    totals = {"개선": 0, "동일": 0, "하락": 0}
    declined = []
    old_sum = 0.0
    new_sum = 0.0

    print("질문 | 기존 순위 → 정제 순위 | RR@3 변화 | 판정")
    print("-" * 65)

    for qid in sorted(old):
        if old[qid]["question"] != new[qid]["question"]:
            raise ValueError(f"{qid}: 질문 내용이 다릅니다.")

        old_rank, old_rr = measure(old[qid])
        new_rank, new_rr = measure(new[qid])
        delta = new_rr - old_rr

        if delta > 1e-9:
            status = "개선"
        elif delta < -1e-9:
            status = "하락"
            declined.append(qid)
        else:
            status = "동일"

        totals[status] += 1
        old_sum += old_rr
        new_sum += new_rr

        old_display = old_rank if old_rank is not None else "없음"
        new_display = new_rank if new_rank is not None else "없음"

        print(
            f"{qid} | {old_display} → {new_display} "
            f"| {delta:+.4f} | {status}"
        )

    count = len(old)
    print(f"\n기존 MRR@3: {old_sum / count:.4f}")
    print(f"정제 MRR@3: {new_sum / count:.4f}")
    print(
        f"개선 {totals['개선']}개 / "
        f"동일 {totals['동일']}개 / "
        f"하락 {totals['하락']}개"
    )

    for qid in declined:
        print(f"\n=== {qid}: 하락 원인 검토 대상 ===")
        print("질문:", old[qid]["question"])
        show_top3("기존", old[qid])
        show_top3("정제", new[qid])

    print("\n각 버전의 초안 라벨 기준입니다.")
    print("관련 라벨이 없다는 것이 실제 근거가 없다는 뜻은 아닙니다.")


if __name__ == "__main__":
    main()