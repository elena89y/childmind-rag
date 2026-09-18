import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("검색할 청크가 없습니다.")

        if not torch.cuda.is_available():
            raise RuntimeError("GPU를 사용할 수 없습니다.")

        self.chunks = chunks
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        print("모델 불러오는 중", flush=True)
        self.model = SentenceTransformer(
            self.model_name,
            device="cuda",
        )

        texts = [chunk["text"] for chunk in chunks]
        self.check_input_lengths(texts, "청크")

        print("청크 임베딩 시작", flush=True)
        self.document_vectors = self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def check_input_lengths(self, texts: list[str], label: str):
        tokenized = self.model.tokenizer(
            texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )

        lengths = [len(ids) for ids in tokenized["input_ids"]]

        print(f"{label} 최대 토큰 수:", max(lengths))

        for index, length in enumerate(lengths):
            if length > self.model.max_seq_length:
                raise ValueError(
                    f"{label} 인덱스 {index}: "
                    f"{length}토큰으로 입력 제한을 초과했습니다."
                )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if not query.strip():
            raise ValueError("질문이 비어 있습니다.")

        if k <= 0:
            raise ValueError("k는 0보다 커야 합니다.")

        self.check_input_lengths([query], "질문")

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.document_vectors @ query_vector

        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:k]

        return [
            {
                "chunk_id": self.chunks[index]["chunk_id"],
                "pdf_page": self.chunks[index]["pdf_page"],
                "text": self.chunks[index]["text"],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_path = (
        project_root / "data" / "processed" / "attachment_chunks.json"
    )

    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    retriever = DenseRetriever(chunks)

    query = "How can caregivers adapt to a child's individual characteristics?"

    print("\n질문:", query)

    for rank, result in enumerate(retriever.retrieve(query), start=1):
        print(f"\n=== 순위 {rank} ===")
        print("청크 ID:", result["chunk_id"])
        print("유사도:", round(result["score"], 4))
        print("본문:", result["text"])