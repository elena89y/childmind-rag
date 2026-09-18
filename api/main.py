import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from evaluation.evaluate_hybrid import reciprocal_rank_fusion
from src.bm25_retriever import tokenize
from src.dense_retriever import DenseRetriever
from src.rag import generate_answer


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="영어로 질문을 입력하세요. 답변은 한국어로 생성됩니다.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = (
        project_root / "data" / "processed" / "attachment_chunks.json"
    )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("검색할 청크가 없습니다.")

    app.state.chunks = chunks
    app.state.chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    app.state.bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    app.state.dense = DenseRetriever(chunks)
    app.state.query_lock = Lock()

    print("RAG 검색기 준비 완료", flush=True)

    yield


app = FastAPI(
    title="ChildMind RAG API",
    description="로컬 Qwen 기반 문헌 검색 및 답변 생성",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "ChildMind RAG API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retriever_ready": True,
        "chunk_count": len(app.state.chunks),
    }


@app.post("/query")
def query_rag(body: QueryRequest):
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="공백만 있는 질문은 입력할 수 없습니다.",
        )

    # 한국어 검색 지원은 별도 작업으로 추가할 예정
    if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", question):
        raise HTTPException(
            status_code=422,
            detail="현재 검색 버전에서는 영어로 질문해주세요.",
        )

    query_tokens = tokenize(question)

    if not query_tokens:
        raise HTTPException(
            status_code=422,
            detail="영어 단어가 포함된 질문을 입력해주세요.",
        )

    # GPU 요청이 동시에 실행되는 것을 방지
    if not app.state.query_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="다른 질문을 처리 중입니다. 잠시 후 다시 시도해주세요.",
        )

    started = time.perf_counter()

    try:
        chunks = app.state.chunks

        # BM25 전체 순위
        scores = app.state.bm25.get_scores(query_tokens)

        indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), index),
        )

        bm25_ids = [
            chunks[index]["chunk_id"]
            for index in indices
        ]

        # Dense 전체 순위
        dense_results = app.state.dense.retrieve(
            question,
            k=len(chunks),
        )

        dense_ids = [
            item["chunk_id"]
            for item in dense_results
        ]

        # RRF 결합 후 상위 3개 근거 선택
        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_constant=60,
        )

        sources = [
            app.state.chunks_by_id[item["chunk_id"]]
            for item in fused[:3]
        ]

        # 로컬 Qwen으로 답변 생성
        response = generate_answer(question, sources)
        answer = response.get("message", {}).get("content", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama가 빈 답변을 반환했습니다.",
            )

        # 인용 번호 확인
        cited_numbers = sorted({
            int(number)
            for number in re.findall(r"\[(\d+)\]", answer)
        })

        valid_numbers = set(range(1, len(sources) + 1))
        invalid_numbers = sorted(
            set(cited_numbers) - valid_numbers
        )

        warnings = []

        if invalid_numbers:
            warnings.append(
                f"존재하지 않는 출처 번호: {invalid_numbers}"
            )

        # 정해진 근거 부족 문장과 정확히 일치하는지 확인
        is_abstention = (
            answer == "제공된 문헌 근거만으로는 답하기 어렵습니다."
        )

        # 일반 답변에서 인용이 누락된 경우에만 경고
        if not cited_numbers and not is_abstention:
            warnings.append(
                "답변에 [숫자] 형태의 출처 인용이 없습니다."
            )

        # 생성 길이 경고는 답변 종류와 관계없이 확인
        if response.get("done_reason") == "length":
            warnings.append(
                "생성 토큰 제한으로 답변이 끊겼을 수 있습니다."
            )

        # 정상 응답은 인용 유무와 관계없이 반환
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "citation_number": number,
                    "chunk_id": source["chunk_id"],
                    "document": source["source_file"],
                    "pdf_page": source["pdf_page"],
                    "text": source["text"],
                }
                for number, source in enumerate(sources, start=1)
            ],
            "cited_numbers": cited_numbers,
            "warnings": warnings,
            "done_reason": response.get("done_reason"),
            "elapsed_seconds": round(
                time.perf_counter() - started,
                2,
            ),
        }

    except HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama가 HTTP {error.code} 오류를 반환했습니다.",
        ) from error

    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Ollama 응답 대기 시간이 초과됐습니다.",
        ) from error

    except URLError as error:
        raise HTTPException(
            status_code=503,
            detail="Ollama에 연결할 수 없습니다. 실행 상태를 확인해주세요.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    finally:
        app.state.query_lock.release()