import torch
from sentence_transformers import CrossEncoder


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"

    print("GPU:", torch.cuda.get_device_name(0))
    print("모델 불러오는 중:", model_name, flush=True)

    model = CrossEncoder(
        model_name,
        device="cuda",
        max_length=512,
    )

    question = "What does responsive caregiving involve?"

    # 동작 확인용 예시 문장이다. 실제 평가 데이터는 아니다.
    passages = [
        "Responsive caregivers notice a child's needs and respond "
        "with physical care, emotional communication, and affection.",

        "Attachment research examines relationships between "
        "children and their caregivers.",

        "The train arrives at the station at nine o'clock.",
    ]

    pairs = [
        (question, passage)
        for passage in passages
    ]

    print("질문과 본문 쌍의 관련성 계산 중...", flush=True)

    scores = model.predict(
        pairs,
        batch_size=3,
        show_progress_bar=False,
    )

    ranked_indices = sorted(
        range(len(passages)),
        key=lambda index: (-float(scores[index]), index),
    )

    print("\n질문:", question)

    for rank, index in enumerate(ranked_indices, start=1):
        print(f"\n=== 순위 {rank} ===")
        print("관련성 점수:", round(float(scores[index]), 4))
        print("본문:", passages[index])


if __name__ == "__main__":
    main()