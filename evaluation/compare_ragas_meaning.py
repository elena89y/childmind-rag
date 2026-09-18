import asyncio
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

# 앞서 만든 모듈의 추적 기능과 평가 설정을 재사용한다.
from evaluation.inspect_ragas_trace import (
    ChatOllama,
    LangchainLLMWrapper,
    RunConfig,
    SingleTurnSample,
    TracedFaithfulness,
)


ANSWERS = {
    "correct": (
        "유아는 언어 능력이 부족하여 자신의 필요를 "
        "돌봐주는 사람에게 표현할 수 없기 때문에 "
        "행동을 통해 자신의 필요를 전달하는 경향이 있다 [1]."
    ),
    "reversed": (
        "유아는 언어 능력이 부족하여 자신의 필요를 "
        "돌봄 받는 사람에게 표현할 수 없기 때문에 "
        "행동을 통해 자신의 필요를 전달하는 경향이 있다 [1]."
    ),
}

LABELS = {
    "correct": "올바른 관계",
    "reversed": "관계가 뒤집힌 답변",
}


async def main():
    if version("ragas") != "0.4.3":
        raise ValueError("이 실험은 RAGAS 0.4.3 기준입니다.")

    root = Path(__file__).resolve().parent.parent
    input_path = (
        root
        / "data"
        / "processed"
        / "chunking_answer_comparisons"
        / "1789734372126824200"
        / "q11_sentence_v2.json"
    )

    record = json.loads(input_path.read_text(encoding="utf-8"))

    question = record["question"]
    contexts = [source["text"] for source in record["sources"]]

    settings = {
        "model": "qwen3:14b",
        "base_url": "http://127.0.0.1:11434",
        "temperature": 0,
        "reasoning": False,
        "num_ctx": 8192,
        "num_predict": 2048,
    }

    model = ChatOllama(
        **settings,
        client_kwargs={"timeout": 300},
    )

    evaluator = LangchainLLMWrapper(
        model,
        run_config=RunConfig(timeout=300, max_retries=1),
    )

    output_dir = (
        root
        / "data"
        / "processed"
        / "ragas_meaning_comparisons"
        / str(time.time_ns())
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results = []

    for case, answer in ANSWERS.items():
        # 사례마다 추적 기록을 새로 만든다.
        metric = TracedFaithfulness(llm=evaluator)
        metric.trace = {}

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        print(f"\n=== {LABELS[case]} ===")
        print("답변:", answer)
        print("평가 중...", flush=True)

        started = time.perf_counter()

        result = {
            "case": case,
            "question": question,
            "answer": answer,
            "retrieved_contexts": contexts,
            "evaluator_settings": settings,
            "ragas_version": version("ragas"),
        }

        try:
            score = float(
                await metric.single_turn_ascore(
                    sample,
                    timeout=600,
                )
            )

            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"유효하지 않은 점수: {score}")

            result["status"] = "ok"
            result["score"] = score

        except Exception as error:
            result["status"] = "error"
            result["error"] = f"{type(error).__name__}: {error}"

        result["trace"] = metric.trace
        result["elapsed_seconds"] = round(
            time.perf_counter() - started,
            3,
        )

        (output_dir / f"{case}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append(result)

        print("\n추출된 주장:")
        print(json.dumps(
            metric.trace.get("statement_extraction"),
            ensure_ascii=False,
            indent=2,
        ))

        print("\n주장별 판정:")
        print(json.dumps(
            metric.trace.get("statement_verdicts"),
            ensure_ascii=False,
            indent=2,
        ))

        if result["status"] == "ok":
            print("\n근거 충실도:", result["score"])
        else:
            print("\n오류:", result["error"])

    print("\n=== 비교 결과 ===")

    for result in results:
        value = result.get("score", result.get("error"))
        print(f"{LABELS[result['case']]}: {value}")

    print("\n저장 폴더:", output_dir)
    print("각 답변을 한 번씩 평가한 결과이며 반복 검증은 아닙니다.")


if __name__ == "__main__":
    asyncio.run(main())