import asyncio
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

# 평가 내용의 외부 추적과 사용 통계 전송을 끈다.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_ollama import ChatOllama
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig


async def main():
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

    if record.get("status") != "ok" or not record.get("answer"):
        raise ValueError("평가할 정상 답변이 없습니다.")

    sample = SingleTurnSample(
        user_input=record["question"],
        response=record["answer"],
        retrieved_contexts=[
            source["text"]
            for source in record["sources"]
        ],
    )

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
        run_config=RunConfig(
            timeout=300,
            max_retries=1,
        ),
    )

    metric = Faithfulness(llm=evaluator)

    print("질문:", record["question"])
    print("평가할 답변:", record["answer"])
    print("\n로컬 모델로 근거 충실도를 평가합니다.", flush=True)

    started = time.perf_counter()

    score = await metric.single_turn_ascore(
        sample,
        timeout=600,
    )
    score = float(score)

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"유효하지 않은 평가 점수: {score}")

    result = {
        "question_id": record["question_id"],
        "version": record["version"],
        "metric": "faithfulness",
        "score": score,
        "evaluator_settings": settings,
        "ragas_version": version("ragas"),
        "langchain_ollama_version": version("langchain-ollama"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_file": str(input_path),
        "sample": sample.model_dump(),
        "note": "연결 확인용 단일 사례이며 독립적인 정답 판정이 아님",
    }

    output_dir = root / "data" / "processed" / "ragas_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.time_ns()}.json"

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n근거 충실도:", score)
    print("소요 시간:", result["elapsed_seconds"], "초")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    asyncio.run(main())