import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

function getErrorMessage(data) {
  if (typeof data?.detail === 'string') {
    return data.detail
  }

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg)
      .filter(Boolean)
      .join(', ')
  }

  return '요청을 처리하지 못했습니다. 입력 내용을 확인해주세요.'
}

function formatConversationTime(value) {
  if (!value) return ''

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return date.toLocaleString('ko-KR', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [conversations, setConversations] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const scrollRef = useRef(null)

  const loadConversations = useCallback(async () => {
    try {
      setHistoryError('')

      const response = await fetch('/api/conversations')
      const data = await response.json()

      if (!response.ok) {
        throw new Error(getErrorMessage(data))
      }

      setConversations(data.conversations)
    } catch (error) {
      setHistoryError(
        error instanceof Error
          ? error.message
          : '대화 기록을 불러오지 못했습니다.',
      )
    } finally {
      setHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, loading])

  async function createConversation() {
    const response = await fetch('/api/conversations', {
      method: 'POST',
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(getErrorMessage(data))
    }

    setActiveConversationId(data.id)

    setConversations((previous) => [
      data,
      ...previous.filter(
        (conversation) => conversation.id !== data.id,
      ),
    ])

    return data.id
  }

  async function handleSubmit(event) {
    event.preventDefault()

    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || loading) {
      return
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmedQuestion,
    }

    setMessages((previous) => [
      ...previous,
      userMessage,
    ])
    setQuestion('')
    setLoading(true)

    try {
      let conversationId = activeConversationId

      if (!conversationId) {
        conversationId = await createConversation()
      }

      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          question: trimmedQuestion,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(getErrorMessage(data))
      }

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          ...data,
        },
      ])

      await loadConversations()
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : '요청에 실패했습니다. 서버 실행 상태를 확인해주세요.'

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          error: message,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleNewChat() {
    if (loading) {
      return
    }

    setActiveConversationId(null)
    setMessages([])
    setQuestion('')
  }

  async function handleConversationSelect(conversationId) {
    if (loading || conversationId === activeConversationId) {
      return
    }

    setLoading(true)
    setHistoryError('')

    try {
      const response = await fetch(
        `/api/conversations/${encodeURIComponent(conversationId)}`,
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(getErrorMessage(data))
      }

      const restoredMessages = data.turns.flatMap((turn) => [
        {
          id: `turn-${turn.id}-user`,
          role: 'user',
          content: turn.question,
        },
        {
          id: `turn-${turn.id}-assistant`,
          role: 'assistant',
          ...turn.response,
        },
      ])

      setActiveConversationId(data.id)
      setMessages(restoredMessages)
      setQuestion('')
    } catch (error) {
      setHistoryError(
        error instanceof Error
          ? error.message
          : '대화 내용을 불러오지 못했습니다.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="layout">
      <aside
        className={`sidebar ${
          sidebarOpen
            ? 'sidebar--open'
            : 'sidebar--closed'
        }`}
      >
        <div className="sidebar-top">
          <button
            type="button"
            className="icon-button"
            aria-label={
              sidebarOpen
                ? '사이드바 닫기'
                : '사이드바 열기'
            }
            onClick={() => {
              setSidebarOpen((open) => !open)
            }}
          >
            <HamburgerIcon />
          </button>

          <span className="sidebar-title">
            CHILDMIND
          </span>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={handleNewChat}
          disabled={loading}
        >
          <PlusIcon />
          새 대화
        </button>

        <div className="sidebar-section">
          <p className="sidebar-section-title">
            대화 기록
          </p>

          {historyLoading ? (
            <div className="history-empty">
              <p>대화 기록을 불러오는 중이에요.</p>
            </div>
          ) : historyError ? (
            <div className="history-empty">
              <p>{historyError}</p>
            </div>
          ) : conversations.length === 0 ? (
            <div className="history-empty">
              <p>아직 대화 기록이 없어요.</p>
              <p className="history-empty-sub">
                질문을 시작하면 여기에 기록이 쌓여요.
              </p>
            </div>
          ) : (
            <ul className="history-list">
              {conversations.map((conversation) => (
                <li key={conversation.id}>
                  <button
                    type="button"
                    className={`history-button ${
                      conversation.id === activeConversationId
                        ? 'history-button--active'
                        : ''
                    }`}
                    onClick={() => {
                      handleConversationSelect(
                        conversation.id,
                      )
                    }}
                    disabled={loading}
                  >
                    <span className="history-title">
                      {conversation.title}
                    </span>

                    <span className="history-date">
                      {formatConversationTime(
                        conversation.updated_at,
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <main className="main">
        <header className="main-header">
          {!sidebarOpen && (
            <button
              type="button"
              className="icon-button"
              aria-label="사이드바 열기"
              onClick={() => setSidebarOpen(true)}
            >
              <HamburgerIcon />
            </button>
          )}

          <div>
            <p className="eyebrow">
              CHILDMIND RAG
            </p>

            <h1>
              아이의 성장과 배움을 이해하는 문헌 탐색
            </h1>
          </div>
        </header>

        <div
          className="chat-scroll"
          ref={scrollRef}
        >
          {messages.length === 0 ? (
            <div className="chat-empty">
              <p className="chat-empty-title">
                무엇이 궁금하세요?
              </p>

              <p className="chat-empty-sub">
                아동 발달·양육·교육에 관한 질문을 입력해 보세요.
                등록된 문헌에서 관련 내용을 찾아 출처와 함께 답변합니다.
              </p>

              <button
                type="button"
                className="suggestion-chip"
                onClick={() => {
                  setQuestion(
                    '반응적인 돌봄은 무엇인가요?',
                  )
                }}
              >
                반응적인 돌봄은 무엇인가요?
              </button>
            </div>
          ) : (
            <ul className="message-list">
              {messages.map((message) => (
                <li
                  key={message.id}
                  className={
                    `message message--${message.role}`
                  }
                >
                  {message.role === 'user' ? (
                    <div className="bubble bubble--user">
                      {message.content}
                    </div>
                  ) : message.error ? (
                    <div
                      className="bubble bubble--error"
                      role="alert"
                    >
                      {message.error}
                    </div>
                  ) : (
                    <div className="bubble bubble--assistant">
                      <p className="answer">
                        {message.answer}
                      </p>

                      <p className="meta">
                        서버 처리 시간:{' '}
                        {message.elapsed_seconds}초
                      </p>

                      {message.warnings?.length > 0 && (
                        <div className="notice">
                          <strong>
                            확인이 필요한 내용
                          </strong>

                          <ul>
                            {message.warnings.map(
                              (warning, index) => (
                                <li key={index}>
                                  {warning}
                                </li>
                              ),
                            )}
                          </ul>
                        </div>
                      )}

                      {message.sources?.length > 0 && (
                        <div className="sources">
                          <p className="sources-title">
                            검색된 근거
                          </p>

                          {message.sources.map(
                            (source) => (
                              <details
                                key={source.chunk_id}
                                className="source"
                              >
                                <summary>
                                  [{source.citation_number}]{' '}
                                  {source.document}
                                  {' · '}PDF{' '}
                                  {source.pdf_page}쪽

                                  {message.cited_numbers?.includes(
                                    source.citation_number,
                                  ) && (
                                    <span className="badge">
                                      답변에서 인용
                                    </span>
                                  )}
                                </summary>

                                <p className="source-text">
                                  {source.text}
                                </p>
                              </details>
                            ),
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </li>
              ))}

              {loading && (
                <li className="message message--assistant">
                  <div
                    className={
                      'bubble bubble--assistant bubble--loading'
                    }
                    role="status"
                  >
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                </li>
              )}
            </ul>
          )}
        </div>

        <form
          className="composer"
          onSubmit={handleSubmit}
        >
          <textarea
            id="question"
            value={question}
            onChange={(event) => {
              setQuestion(event.target.value)
            }}
            onKeyDown={(event) => {
              if (
                event.key === 'Enter'
                && !event.shiftKey
              ) {
                event.preventDefault()
                handleSubmit(event)
              }
            }}
            placeholder={
              '아이의 발달과 배움에 대해 무엇이 궁금한가요?'
            }
            maxLength={1000}
            rows={2}
            required
            disabled={loading}
          />

          <div className="composer-footer">
            <span className="char-count">
              {question.length} / 1000
            </span>

            <button
              type="submit"
              className="send-button"
              disabled={
                loading || !question.trim()
              }
            >
              {loading
                ? '답변 생성 중…'
                : '질문하기'}
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}

function HamburgerIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M2.5 5.5H17.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M2.5 10H17.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M2.5 14.5H17.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  )
}

function PlusIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M8 2.5V13.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M2.5 8H13.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default App