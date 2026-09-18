import asyncio
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_ollama import ChatOllama
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig


class TracedFaithfulness(Faithfulness):
    # RAGAS 0.4.3의 내부 단계에서 반환되는 결과를 기록한다.
    # 평가 프롬프트와 점수 계산은 부모 클래스의 동작을 그대로 쓴다.
    async def _create_statements(self, row, callbacks):
        result = await super()._create_statements(row, callbacks)
        self.trace["statement_extraction"] = result.model_dump()
        return result

    async def _create_verdicts(self, row, statements, callbacks):
        result = await super()._create_verdicts(
            row, statements, callbacks
        )
        self.trace["statement_verdicts"] = result.model_dump()
        return result


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
        run_config=RunConfig(timeout=300, max_retries=1),
    )

    metric = TracedFaithfulness(llm=evaluator)
    metric.trace = {}

    output_dir = root / "data" / "processed" / "ragas_trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.time_ns()}.json"

    result = {
        "question_id": record["question_id"],
        "version": record["version"],
        "metric": "faithfulness",
        "evaluator_settings": settings,
        "ragas_version": version("ragas"),
        "langchain_ollama_version": version("langchain-ollama"),
        "input_file": str(input_path),
        "sample": sample.model_dump(),
    }

    print("평가할 답변:", record["answer"])
    print("\n주장 추출과 판정 결과를 기록합니다.", flush=True)

    started = time.perf_counter()

    try:
        score = float(
            await metric.single_turn_ascore(sample, timeout=600)
        )

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"유효하지 않은 평가 점수: {score}")

        result["status"] = "ok"
        result["score"] = score

    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"

    finally:
        result["trace"] = metric.trace
        result["elapsed_seconds"] = round(
            time.perf_counter() - started, 3
        )

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n=== 추출된 주장 ===")
    print(json.dumps(
        metric.trace.get("statement_extraction"),
        ensure_ascii=False,
        indent=2,
    ))

    print("\n=== 주장별 판정 ===")
    print(json.dumps(
        metric.trace.get("statement_verdicts"),
        ensure_ascii=False,
        indent=2,
    ))

    if result["status"] == "ok":
        print("\n근거 충실도:", result["score"])
    else:
        print("\n평가 오류:", result["error"])

    print("저장 위치:", output_path)


if __name__ == "__main__":
    asyncio.run(main())