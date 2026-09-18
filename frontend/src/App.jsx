import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault()

    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || loading) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion }),
      })

      const data = await response.json()

      if (!response.ok) {
        const message =
          typeof data.detail === 'string'
            ? data.detail
            : '요청을 처리하지 못했습니다. 입력 내용을 확인해주세요.'
        throw new Error(message)
      }

      setResult(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : '요청에 실패했습니다. 서버 실행 상태를 확인해주세요.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <header>
        <p className="eyebrow">CHILDMIND RAG</p>
        <h1>문헌에서 찾는 양육의 근거</h1>
        <p className="intro">
          아동 발달·양육 문헌을 검색하고, 근거와 함께 답변합니다.
          한국어 또는 영어로 질문하면 한국어로 답변합니다.
        </p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <label htmlFor="question">궁금한 내용을 질문하세요</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="반응적인 돌봄은 무엇인가요?"
          maxLength={1000}
          rows={4}
          required
          disabled={loading}
        />

        <div className="form-footer">
          <span>{question.length} / 1000</span>
          <button
            type="submit"
            disabled={loading || !question.trim()}
          >
            {loading ? '답변 생성 중…' : '질문하기'}
          </button>
        </div>
      </form>

      {loading && (
        <p className="status" role="status">
          문헌을 검색하고 로컬 모델이 답변을 작성하고 있어요.
        </p>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <section className="card" aria-label="질문에 대한 답변">
          <p className="asked-question">{result.question}</p>
          <h2>답변</h2>
          <p className="answer">{result.answer}</p>

          <p className="meta">
            서버 처리 시간: {result.elapsed_seconds}초
          </p>

          {result.warnings?.length > 0 && (
            <div className="notice">
              <strong>확인이 필요한 내용</strong>
              <ul>
                {result.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          <h2>검색된 근거</h2>
          <p className="meta">
            답변에서 인용한 근거에는 ‘답변에서 인용’ 표시가 붙습니다.
          </p>

          {result.sources.map((source) => (
            <details key={source.chunk_id} className="source">
              <summary>
                [{source.citation_number}] {source.document}
                {' · '}PDF {source.pdf_page}쪽
                {result.cited_numbers?.includes(source.citation_number) && (
                  <span className="badge">답변에서 인용</span>
                )}
              </summary>
              <p className="source-text">{source.text}</p>
            </details>
          ))}
        </section>
      )}
    </main>
  )
}

export default App