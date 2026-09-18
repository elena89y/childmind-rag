import json
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

questions = json.loads(
    (project_root / "evaluation" / "questions.json").read_text(
        encoding="utf-8"
    )
)

chunks = json.loads(
    (
        project_root
        / "data"
        / "processed"
        / "attachment_chunks.json"
    ).read_text(encoding="utf-8")
)

# 후보를 찾기 위한 구절이며, 정답 판정 규칙은 아님
anchors = {
    "q01": ["securely attached", "base of operations"],
    "q02": ["strange situation", "mother returned", "20 minutes"],
    "q03": ["responsive mothers", "responsive caregiving"],
    "q04": ["goodness-of-fit", "compatibility"],
    "q05": ["modify their behavior", "fit the needs"],
}


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


report = []

for question in questions:
    question_id = question["question_id"]

    report.append(f"\n=== {question_id} ===")
    report.append(question["question"])

    candidates = []

    for chunk in chunks:
        if chunk["document_id"] != question["document_id"]:
            continue

        text = normalize(chunk["text"])

        matched = [
            phrase
            for phrase in anchors[question_id]
            if normalize(phrase) in text
        ]

        if matched:
            candidates.append(chunk)
            report.append(
                f"\n청크: {chunk['chunk_id']} | "
                f"PDF {chunk['pdf_page']}쪽"
            )
            report.append(f"일치 구절: {', '.join(matched)}")
            report.append(chunk["text"])

    print(f"\n{question_id}: 후보 {len(candidates)}개")

    for chunk in candidates:
        print(f"  {chunk['chunk_id']} | PDF {chunk['pdf_page']}쪽")

    if not candidates:
        report.append("후보 없음: 원문 또는 다른 표현을 확인해야 합니다.")

output_path = (
    project_root / "data" / "processed" / "evidence_candidates.txt"
)

output_path.write_text("\n".join(report), encoding="utf-8")
print("\n후보 본문 저장:", output_path)