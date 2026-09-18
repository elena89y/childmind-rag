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
from src.rag import generate_answer, translate_query


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="한국어 또는 영어로 질문하세요. 답변은 한국어로 생성됩니다.",
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
    version="0.3.0",
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

    if not re.search(r"[a-zA-Z가-힣]", question):
        raise HTTPException(
            status_code=422,
            detail="한국어 또는 영어 단어가 포함된 질문을 입력해주세요.",
        )

    # 번역부터 답변 생성까지 GPU 요청을 순차 처리
    if not app.state.query_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="다른 질문을 처리 중입니다. 잠시 후 다시 시도해주세요.",
        )

    started = time.perf_counter()

    try:
        # 한국어 질문은 영어로 번역하고 영어 질문은 그대로 사용
        try:
            search_query = translate_query(question)
        except ValueError as error:
            raise HTTPException(
                status_code=502,
                detail=f"질문 번역 실패: {error}",
            ) from error

        query_tokens = tokenize(search_query)

        if not query_tokens:
            raise HTTPException(
                status_code=422,
                detail="검색할 영어 단어를 찾지 못했습니다.",
            )

        chunks = app.state.chunks

        # BM25와 Dense 모두 같은 영어 검색 질문 사용
        scores = app.state.bm25.get_scores(query_tokens)
        indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(scores[index]), index),
        )
        bm25_ids = [
            chunks[index]["chunk_id"]
            for index in indices
        ]

        dense_results = app.state.dense.retrieve(
            search_query,
            k=len(chunks),
        )
        dense_ids = [
            item["chunk_id"]
            for item in dense_results
        ]

        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_constant=60,
        )
        sources = [
            app.state.chunks_by_id[item["chunk_id"]]
            for item in fused[:3]
        ]

        # 답변 생성에는 사용자의 원래 질문을 전달
        response = generate_answer(question, sources)
        answer = response.get("message", {}).get("content", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama가 빈 답변을 반환했습니다.",
            )

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

        is_abstention = (
            answer == "제공된 문헌 근거만으로는 답하기 어렵습니다."
        )

        if not cited_numbers and not is_abstention:
            warnings.append(
                "답변에 [숫자] 형태의 출처 인용이 없습니다."
            )

        if response.get("done_reason") == "length":
            warnings.append(
                "생성 토큰 제한으로 답변이 끊겼을 수 있습니다."
            )

        return {
            "question": question,
            "search_query": search_query,
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