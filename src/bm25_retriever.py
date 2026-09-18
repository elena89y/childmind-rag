import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


if __name__ == "__main__":
    sample = "Parenting, attachment and children."
    tokens = tokenize(sample)

    print(tokens)

    assert tokens == [
        "parenting",
        "attachment",
        "and",
        "children",
    ]

    print("토큰 분리 확인 완료")

    project_root = Path(__file__).resolve().parent.parent
    input_path = (
        project_root / "data" / "processed" / "attachment_chunks.json"
    )

    chunks = json.loads(input_path.read_text(encoding="utf-8"))

    tokenized_chunks = []

    for chunk in chunks:
        tokenized_chunks.append(tokenize(chunk["text"]))

    bm25 = BM25Okapi(tokenized_chunks)

    query = "What is secure attachment?"
    original_query_tokens = tokenize(query)

    remove_query_stopwords = False

    if remove_query_stopwords:
        query_tokens = [
            token
            for token in original_query_tokens
            if token not in {"what", "is"}
        ]
    else:
        query_tokens = original_query_tokens.copy()

    print("\n질문:", query)
    print("변경 전 질문 토큰:", original_query_tokens)
    print("변경 후 질문 토큰:", query_tokens)
    print("검색 대상 청크 수:", len(chunks))

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )

    for rank, index in enumerate(ranked_indices[:3], start=1):
        chunk = chunks[index]

        print(f"\n=== 순위 {rank} ===")
        print("청크 ID:", chunk["chunk_id"])
        print("PDF 페이지:", chunk["pdf_page"])
        print("BM25 점수:", round(float(scores[index]), 4))
        print("질문 단어별 기여:")

        for token in query_tokens:
            token_score = bm25.get_scores([token])[index]
            count = tokenized_chunks[index].count(token)

            print(
                f"  {token}: "
                f"등장 {count}회, "
                f"점수 {float(token_score):.4f}"
            )

        print("본문:", chunk["text"])