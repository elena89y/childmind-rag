import torch
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU를 사용할 수 없습니다.")

    model_name = "sentence-transformers/all-mpnet-base-v2"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("모델 불러오는 중:", model_name, flush=True)

    model = SentenceTransformer(
        model_name,
        device="cuda",
    )

    print("모델 장치:", model.device, flush=True)
    print("모델 입력 제한:", model.max_seq_length, flush=True)

    sentences = [
        "Parents adapt their care to the child's needs.",
        "Caregivers adjust their behavior for each child.",
        "The train arrives at the station.",
    ]

    print("문장 3개 임베딩 시작", flush=True)

    vectors = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("벡터 배열 크기:", vectors.shape)
    print("첫 문장 벡터의 앞 5개 값:", vectors[0][:5])

    similarities = vectors @ vectors.T

    print("\n첫 번째 문장과의 유사도:")

    for index, sentence in enumerate(sentences):
        print(f"{similarities[0, index]:.4f} | {sentence}")