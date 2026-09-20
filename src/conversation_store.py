import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "conversations.sqlite3"
)


def connect_database():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def current_time():
    return datetime.now(timezone.utc).isoformat()


def initialize_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(connect_database()) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_turns_conversation_id
                ON turns(conversation_id, id)
            """)


def create_conversation():
    now = current_time()

    conversation = {
        "id": str(uuid4()),
        "title": "새 대화",
        "created_at": now,
        "updated_at": now,
    }

    with closing(connect_database()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, title, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation["id"],
                    conversation["title"],
                    conversation["created_at"],
                    conversation["updated_at"],
                ),
            )

    return conversation


def list_conversations():
    with closing(connect_database()) as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with closing(connect_database()) as connection:
        # 대화 정보와 질문 목록을 같은 읽기 트랜잭션에서 조회한다.
        connection.execute("BEGIN")

        conversation = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            return None

        rows = connection.execute(
            """
            SELECT id, question, response_json, created_at
            FROM turns
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    result = dict(conversation)
    result["turns"] = [
        {
            "id": row["id"],
            "question": row["question"],
            "response": json.loads(row["response_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]

    return result


def save_turn(conversation_id, response):
    question = response.get("question")
    answer = response.get("answer")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("저장할 질문이 비어 있습니다.")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("저장할 답변이 비어 있습니다.")

    question = question.strip()
    response_json = json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
    )

    with closing(connect_database()) as connection:
        with connection:
            # 확인과 저장 사이에 다른 쓰기가 끼어들지 않도록 한다.
            connection.execute("BEGIN IMMEDIATE")

            conversation = connection.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if conversation is None:
                raise LookupError("존재하지 않는 대화입니다.")

            has_turn = connection.execute(
                """
                SELECT 1 FROM turns
                WHERE conversation_id = ?
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()

            now = current_time()

            cursor = connection.execute(
                """
                INSERT INTO turns (
                    conversation_id,
                    question,
                    response_json,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    question,
                    response_json,
                    now,
                ),
            )
            turn_id = cursor.lastrowid

            if has_turn is None:
                # 첫 질문을 대화 제목으로 사용한다.
                title = " ".join(question.split())

                if len(title) > 40:
                    title = title[:40] + "…"

                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (title, now, conversation_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (now, conversation_id),
                )

    return {
        "id": turn_id,
        "conversation_id": conversation_id,
        "question": question,
        "response": json.loads(response_json),
        "created_at": now,
    }


if __name__ == "__main__":
    initialize_database()
    print("대화 저장 기능 준비 완료")
    print("저장 위치:", DB_PATH)
    print("저장된 대화 수:", len(list_conversations()))