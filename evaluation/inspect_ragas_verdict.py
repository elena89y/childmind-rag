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


async def main():
    if version("ragas") != "0.4.3":
        raise ValueError("이 검사 코드는 RAGAS 0.4.3용입니다.")

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
    metric = TracedFaithfulness(llm=evaluator)
    metric.trace = {}

    # 기존 추출 결과와 같은 영어로 관계만 구분한다.
    cases = [
        {
            "name": "올바른 관계",
            "statement": (
                "Young children are unable to verbally express "
                "their needs to people who provide care."
            ),
        },
        {
            "name": "관계가 뒤집힌 주장",
            "statement": (
                "Young children are unable to verbally express "
                "their needs to people who receive care."
            ),
        },
    ]

    row = {
        "retrieved_contexts": [
            source["text"]
            for source in record["sources"]
        ]
    }

    output_dir = (
        root / "data" / "processed"
        / "ragas_verdict_checks" / str(time.time_ns())
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    for case in cases:
        print(f"\n=== {case['name']} ===", flush=True)

        # 주장 추출 없이 근거 대조 단계에 직접 전달한다.
        verdicts = await asyncio.wait_for(
            metric._create_verdicts(
                row,
                [case["statement"]],
                callbacks=[],
            ),
            timeout=600,
        )

        result = verdicts.model_dump()
        case["result"] = result

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ))

        # 각 호출이 끝날 때 저장한다.
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "settings": settings,
                    "ragas_version": version("ragas"),
                    "contexts": row["retrieved_contexts"],
                    "cases": cases,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n저장 위치:", output_dir / "results.json")


if __name__ == "__main__":
    asyncio.run(main())