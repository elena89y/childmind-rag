import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


if __name__ == "__main__":
    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "qwen3:14b",
        "messages": [
            {
                "role": "user",
                "content": "한국어로 '로컬 모델 연결 성공'이라고만 답하세요.",
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 4096,
            "num_predict": 128,
            "temperature": 0,
        },
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("Qwen3 14B에 로컬 요청 중...", flush=True)
    print("첫 요청은 모델을 메모리에 불러오는 시간이 필요합니다.", flush=True)

    started = time.perf_counter()

    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        print("HTTP 오류:", error.code)
        print(error.read().decode("utf-8", errors="replace"))
        raise SystemExit(1)

    except URLError as error:
        print("Ollama 연결 실패:", error.reason)
        raise SystemExit(1)

    answer = result.get("message", {}).get("content", "")

    print("\n답변:", answer)
    print("완료 여부:", result.get("done"))
    print("종료 이유:", result.get("done_reason"))
    print("소요 시간:", round(time.perf_counter() - started, 2), "초")