import asyncio
import json
import time
from importlib.metadata import version
from pathlib import Path

from evaluation.inspect_ragas_trace import (
    ChatOllama,
    LangchainLLMWrapper,
    RunConfig,
    TracedFaithfulness,
)


MODELS = [
    "qwen3:14b",
    "gemma3:12b",
    "exaone3.5:7.8b",
]

CASES = [
    {
        "case": "correct",
        "name": "올바른 관계",
        "expected_verdict": 1,
        "statement": (
            "Young children are unable to verbally express "
            "their needs to people who provide care."
        ),
    },
    {
        "case": "reversed",
        "name": "관계가 뒤집힌 주장",
        "expected_verdict": 0,
        "statement": (
            "Young children are unable to verbally express "
            "their needs to people who receive care."
        ),
    },
]


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
    contexts = [source["text"] for source in record["sources"]]

    output_dir = (
        root
        / "data"
        / "processed"
        / "judge_model_comparisons"
        / str(time.time_ns())
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results = []

    print("세 모델에서 두 주장씩 순서대로 평가합니다.")
    print("저장 폴더:", output_dir, flush=True)

    for model_name in MODELS:
        settings = {
            "model": model_name,
            "base_url": "http://127.0.0.1:11434",
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 2048,
            "keep_alive": 0,
        }

        # 추론 기능을 지원하는 Qwen에서만 명시적으로 끈다.
        if model_name == "qwen3:14b":
            settings["reasoning"] = False

        model = ChatOllama(
            **settings,
            client_kwargs={"timeout": 300},
        )
        evaluator = LangchainLLMWrapper(
            model,
            run_config=RunConfig(timeout=300, max_retries=1),
        )

        for case in CASES:
            metric = TracedFaithfulness(llm=evaluator)
            metric.trace = {}

            print(
                f"\n=== {model_name} | {case['name']} ===",
                flush=True,
            )

            started = time.perf_counter()
            result = {
                "model": model_name,
                **case,
                "settings": settings,
            }

            try:
                # 기대 판정값은 모델에 전달하지 않는다.
                response = await asyncio.wait_for(
                    metric._create_verdicts(
                        {"retrieved_contexts": contexts},
                        [case["statement"]],
                        callbacks=[],
                    ),
                    timeout=600,
                )

                parsed = response.model_dump()
                result["response"] = parsed
                statements = parsed["statements"]

                if len(statements) != 1:
                    raise ValueError("판정할 주장 수가 달라졌습니다.")

                judged = statements[0]

                if judged["statement"].strip() != case["statement"]:
                    raise ValueError(
                        "판정 출력에서 입력 주장이 변경됐습니다."
                    )

                verdict = judged["verdict"]

                if verdict not in (0, 1):
                    raise ValueError("판정값이 0 또는 1이 아닙니다.")

                result.update({
                    "status": "ok",
                    "verdict": verdict,
                    "reason": judged["reason"],
                    "matches_expected": (
                        verdict == case["expected_verdict"]
                    ),
                })

                print("판정:", verdict)
                print("이유:", judged["reason"])

            except Exception as error:
                result.update({
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
                print("오류:", result["error"])

            result["trace"] = metric.trace
            result["elapsed_seconds"] = round(
                time.perf_counter() - started, 3
            )
            results.append(result)

            # 한 사례가 끝날 때마다 저장한다.
            (output_dir / "results.json").write_text(
                json.dumps(
                    {
                        "ragas_version": version("ragas"),
                        "langchain_ollama_version": version(
                            "langchain-ollama"
                        ),
                        "input_file": str(input_path),
                        "contexts": contexts,
                        "note": (
                            "고정된 영어 주장 두 개의 판정 실험. "
                            "사례별 1회이며 일반 성능 평가는 아님."
                        ),
                        "results": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print("\n=== 모델별 비교 ===")
    print("모델 | 사례 | 기대 판정 | 실제 판정")

    for result in results:
        actual = (
            result["verdict"]
            if result["status"] == "ok"
            else "오류"
        )
        print(
            f"{result['model']} | {result['name']} | "
            f"{result['expected_verdict']} | {actual}"
        )

    print("\n저장 위치:", output_dir / "results.json")


if __name__ == "__main__":
    asyncio.run(main())