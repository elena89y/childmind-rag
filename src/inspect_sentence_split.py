import re
import json
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())

    if not normalized:
        return []

    abbreviations = {
        "dr.", "mr.", "mrs.", "ms.", "prof.",
        "fig.", "figs.", "e.g.", "i.e.", "al.",
    }

    # 문장 끝 기호 다음에 닫는 따옴표·괄호가 있어도 경계로 찾는다.
    boundary = re.compile(r"""[.!?]["'”’)]*\s+""")

    parts = []
    start = 0

    for match in boundary.finditer(normalized):
        parts.append(normalized[start:match.end()].strip())
        start = match.end()

    if start < len(normalized):
        parts.append(normalized[start:])

    sentences = []
    pending = ""

    for part in parts:
        pending = f"{pending} {part}".strip()

        # 약어 판정 때만 앞의 괄호·따옴표를 제외한다.
        # 실제 본문은 바꾸지 않는다.
        last_word = pending.split()[-1].lstrip("([\"'“‘").lower()

        if last_word in abbreviations:
            continue

        sentences.append(pending)
        pending = ""

    if pending:
        sentences.append(pending)

    return sentences


def main():
    sample = (
        "The mother left the room. "
        "The infant became upset. "
        "What happened when she returned? "
        "The infant calmed down."
    )

    sentences = split_sentences(sample)

    for number, sentence in enumerate(sentences, start=1):
        print(f"{number}번 문장: {sentence}")

    assert len(sentences) == 4
    assert " ".join(sentences) == " ".join(sample.split())
    assert split_sentences("   ") == []

    print("\n기본 문장 분리와 본문 보존 확인 완료")

    # 단순한 규칙이 잘못 나누는 경우도 확인한다.
    abbreviation_sample = "Dr. Smith described attachment."
    print("\n약어가 있는 문장:")
    for number, sentence in enumerate(
        split_sentences(abbreviation_sample),
        start=1,
    ):
        print(f"{number}번 문장: {sentence}")

def inspect_paper():
    root = Path(__file__).resolve().parent.parent
    input_path = (
        root / "data" / "processed"
        / "attachment_pages_sections.json"
    )
    pages = json.loads(input_path.read_text(encoding="utf-8"))

    total_chunks = 0

    for page in pages:
        for section in page["sections"]:
            if not section["searchable"]:
                continue

            text = section["text"]
            sentences = split_sentences(text)

            assert " ".join(sentences) == " ".join(text.split())

            chunks = group_sentences(
                sentences,
                target_words=200,
                overlap=True,
            )
            total_chunks += len(chunks)

            print(
                f"\n=== PDF {page['pdf_page']}쪽 "
                f"| {section['kind']} | {len(chunks)}개 청크 ==="
            )

            for number, chunk in enumerate(chunks, start=1):
                words = chunk.split()

                print(f"\n청크 {number} | {len(words)}단어")
                print("시작:", " ".join(words[:20]))
                print("끝:", " ".join(words[-20:]))

                if len(words) > 200:
                    print(
                        "확인 필요: 긴 문장 또는 분리되지 않은 "
                        "영역이 200단어를 넘었습니다."
                    )

    print("\n전체 청크 수:", total_chunks)
    print("화면 출력만 완료했습니다. 파일은 저장하지 않았습니다.")


def group_sentence_records(
    sentences: list[str],
    target_words: int = 200,
    overlap: bool = False,
) -> list[dict]:
    if target_words <= 0:
        raise ValueError("목표 단어 수는 양수여야 합니다.")

    records = []
    start = 0
    previous_end = 0

    while start < len(sentences):
        end = start
        word_count = 0

        while end < len(sentences):
            next_words = len(sentences[end].split())

            # 첫 문장은 길더라도 온전히 담는다.
            if end > start and word_count + next_words > target_words:
                break

            word_count += next_words
            end += 1

        records.append({
            "text": " ".join(sentences[start:end]),
            "sentence_start": start,
            "sentence_end": end,
            "overlap_sentences": max(0, previous_end - start),
            "word_count": word_count,
        })

        previous_end = end

        if end == len(sentences):
            break

        last_words = len(sentences[end - 1].split())
        next_words = len(sentences[end].split())

        if (
            overlap
            and end - start >= 2
            and last_words + next_words <= target_words
        ):
            start = end - 1
        else:
            start = end

    # 겹친 문장을 제외하고 원래 문장 목록을 복원한다.
    reconstructed = []

    for record in records:
        new_start = (
            record["sentence_start"]
            + record["overlap_sentences"]
        )
        reconstructed.extend(
            sentences[new_start:record["sentence_end"]]
        )

    if reconstructed != sentences:
        raise ValueError("겹침을 제외한 문장 복원에 실패했습니다.")

    return records


def group_sentences(
    sentences: list[str],
    target_words: int = 200,
    overlap: bool = False,
) -> list[str]:
    records = group_sentence_records(
        sentences,
        target_words=target_words,
        overlap=overlap,
    )

    return [record["text"] for record in records]

def inspect_grouping():
    sentences = [
        "The mother left.",        # 3단어
        "The infant cried.",       # 3단어
        "The mother returned.",    # 3단어
        "The infant calmed down.", # 4단어
    ]

    # 1. 겹침 없이 묶기
    chunks = group_sentences(sentences, target_words=6)

    print("\n=== 문장 묶기 예제 ===")
    for number, chunk in enumerate(chunks, start=1):
        print(f"{number}번 | {len(chunk.split())}단어 | {chunk}")

    assert chunks == [
        "The mother left. The infant cried.",
        "The mother returned.",
        "The infant calmed down.",
    ]
    assert " ".join(chunks) == " ".join(sentences)

    print("문장 묶기와 본문 보존 확인 완료")

    # 2. 마지막 문장 하나를 겹쳐서 묶기
    overlapping_chunks = group_sentences(
        sentences,
        target_words=9,
        overlap=True,
    )

    print("\n=== 한 문장 겹침 예제 ===")
    for number, chunk in enumerate(overlapping_chunks, start=1):
        print(f"{number}번 | {len(chunk.split())}단어 | {chunk}")

    assert overlapping_chunks == [
        "The mother left. The infant cried. The mother returned.",
        "The mother returned. The infant calmed down.",
    ]

    print("한 문장 겹침 확인 완료")

    records = group_sentence_records(
        sentences,
        target_words=9,
        overlap=True,
    )

    assert [
        (
            record["sentence_start"],
            record["sentence_end"],
            record["overlap_sentences"],
        )
        for record in records
    ] == [
        (0, 3, 0),
        (2, 4, 1),
    ]

    print("\n=== 문장 위치 확인 ===")
    for record in records:
        print(
            f"범위 [{record['sentence_start']}, "
            f"{record['sentence_end']}) | "
            f"겹침 {record['overlap_sentences']}문장"
        )

    print("문장 위치·겹침·복원 검사 통과")



if __name__ == "__main__":
    main()
    inspect_paper()
    inspect_grouping()