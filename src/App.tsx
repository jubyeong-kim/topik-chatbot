// 화면 전체. (PRD v3 §5 · §10 · §11)
//
// 한 질문의 흐름: 검색 → 1층 범위 규칙 → 2층 게이트 → 답변 스트리밍 → 판정
// 각 단계를 버튼 라벨로 드러내고, 어느 단계에서 멈췄는지 사용자가 읽을 수 있게 한다.

import { useEffect, useRef, useState } from 'react'
import { load, type LoadProgress, type Store } from './lib/embed'
import { buildBm25Index, search, type Hit, type SearchResult } from './lib/search'
import { answer, gate, health, ANSWER_K, GATE_K, type Health } from './lib/ollama'
import { judge, type JudgeResult } from './lib/judge'
import { load as loadFeedback, save as saveFeedback, summary as feedbackSummary } from './lib/feedback'

const EXAMPLES = [
  '54번은 어떤 기준으로 채점되나요?',
  '53번에서 감점되는 경우는 뭔가요?',
  '3급과 5급의 평가 기준은 뭐가 다른가요?',
  'TOPIK 응시료는 얼마인가요?',
]

const LIMITS = [
  '학습자 개인 작문을 채점해주는 기능은 아닙니다.',
  '수집된 자료에 없는 내용은 답하지 않습니다.',
  'PBT 기준과 IBT 기준이 다릅니다 — 답변에 어느 쪽인지 표시됩니다.',
]

type Ready = { store: Store; embed: (t: string) => Promise<number[]>; index: ReturnType<typeof buildBm25Index> }

export default function App() {
  const [progress, setProgress] = useState<LoadProgress | null>(null)
  const [ready, setReady] = useState<Ready | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState<false | '검색' | '판단' | '작성' | '판정'>(false)
  const [result, setResult] = useState<SearchResult | null>(null)
  const [reply, setReply] = useState('')
  const [refusal, setRefusal] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<JudgeResult | null>(null)
  const [srv, setSrv] = useState<Health | null>(null)
  const [openChunk, setOpenChunk] = useState<Hit | null>(null)   // 근거 모달
  const [rated, setRated] = useState(false)                       // 이 답변에 피드백했는가
  const [fb, setFb] = useState(() => feedbackSummary(loadFeedback()))
  const abort = useRef<AbortController | null>(null)

  useEffect(() => { void health().then(setSrv) }, [])

  async function ensureReady(): Promise<Ready> {
    if (ready) return ready
    const { store, embed } = await load(import.meta.env.BASE_URL, setProgress)
    const r = { store, embed, index: buildBm25Index(store.chunks) }
    setReady(r)
    return r
  }

  async function run(text: string) {
    if (!text.trim() || busy) return
    setError(null); setResult(null); setReply(''); setRefusal(null); setVerdict(null); setRated(false)
    const ac = new AbortController()
    abort.current = ac

    try {
      setBusy('검색')
      const r = await ensureReady()
      const found = search(r.store.chunks, r.index, text, await r.embed(text))
      setResult(found)

      // 1층에서 걸렸으면 여기서 끝
      if (found.verdict === 'refuse') {
        setRefusal(found.reason)
        return
      }

      // 2층 — 이 자료로 답할 수 있는지 모델에게 묻는다
      setBusy('판단')
      // 게이트는 짧게, 생성은 넉넉히. 지연의 대부분이 프롬프트 읽기라 게이트를 줄이면
      // 체감 대기가 절반이 되고, 생성은 8개를 다 받아 근거를 놓치지 않는다.
      const forGate = found.hits.slice(0, GATE_K)
      const forAnswer = found.hits.slice(0, ANSWER_K)
      const g = await gate(text, forGate, { signal: ac.signal })
      if (!g.answerable) {
        setRefusal('검색된 자료에 이 질문의 답이 없습니다')
        return
      }

      setBusy('작성')
      let full = ''
      for await (const piece of answer(text, forAnswer, { signal: ac.signal, weak: found.verdict === 'weak' })) {
        full += piece
        setReply(full)
      }

      // 판정은 답변이 끝난 뒤. 실패해도 답변과 출처는 그대로 둔다.
      setBusy('판정')
      setVerdict(await judge(text, forAnswer, full, { signal: ac.signal }))
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') setReply((s) => s + '\n\n(중단됨)')
      else setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      abort.current = null
    }
  }

  return (
    <main>
      <h1 style={{ fontSize: 28, marginBottom: 4 }}>TOPIK 채점 안내봇</h1>
      <p style={{ color: 'var(--muted)', marginTop: 0 }}>
        TOPIK 쓰기 채점 기준을 공개 자료에 근거해 안내합니다.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          void run(q)
        }}
        style={{ display: 'flex', gap: 8, marginTop: 24 }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="TOPIK 쓰기 채점에 대해 물어보세요"
          style={{ flex: 1, padding: '10px 12px', fontSize: 15, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--bg)', color: 'var(--fg)' }}
        />
        <button type="submit" disabled={!!busy} style={{ padding: '10px 18px', fontSize: 15, borderRadius: 8, border: '1px solid var(--line)', cursor: busy ? 'wait' : 'pointer' }}>
          {busy ? `${busy} 중…` : '물어보기'}
        </button>
        {busy && (
          <button type="button" onClick={() => abort.current?.abort()} style={{ padding: '10px 14px', fontSize: 15, borderRadius: 8, border: '1px solid var(--line)', cursor: 'pointer' }}>
            중단
          </button>
        )}
      </form>

      {srv && !srv.ok && (
        <p style={{ marginTop: 14, padding: '10px 14px', borderRadius: 8, background: 'var(--warn-bg)', color: 'var(--warn)', fontSize: 14 }} role="alert">
          {srv.reason}{' '}
          <button
            onClick={() => { setSrv(null); void health().then(setSrv) }}
            style={{ marginLeft: 6, padding: '3px 10px', fontSize: 13, borderRadius: 6, border: '1px solid var(--line)', background: 'transparent', color: 'inherit', cursor: 'pointer' }}
          >
            다시 확인
          </button>
        </p>
      )}
      {srv?.ok && !srv.hasModel && (
        <p style={{ marginTop: 14, color: 'var(--warn)', fontSize: 14 }} role="alert">
          Ollama 는 실행 중이지만 모델이 없습니다. <code>ollama pull qwen2.5:1.5b-instruct-q4_K_M</code>
        </p>
      )}

      <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {EXAMPLES.map((x) => (
          <button
            key={x}
            onClick={() => { setQ(x); void run(x) }}
            style={{ fontSize: 13, padding: '5px 10px', borderRadius: 999, border: '1px solid var(--line)', background: 'transparent', color: 'var(--muted)', cursor: 'pointer' }}
          >
            {x}
          </button>
        ))}
      </div>

      {progress && !ready && (
        <p style={{ marginTop: 20, color: 'var(--muted)', fontSize: 14 }}>
          {progress.message}
          {progress.phase === 'model' && ` ${progress.percent}%`}
        </p>
      )}

      {error && (
        <p style={{ marginTop: 20, color: 'var(--warn)', fontSize: 14 }} role="alert">
          오류: {error}
        </p>
      )}

      {ready && progress?.phase === 'ready' && (
        <p style={{ marginTop: 16, fontSize: 12, color: 'var(--muted)' }}>
          {ready.store.model} · {ready.store.dtype} · {ready.store.dim}차원 · 청크 {ready.store.chunks.length}개 ·
          {progress.cached ? ' 캐시 사용' : ' 최초 다운로드'} · 준비 {(progress.elapsedMs / 1000).toFixed(1)}초
        </p>
      )}

      {refusal && (
        <section style={{ marginTop: 24, padding: 16, border: '1px solid var(--line)', borderRadius: 10 }}>
          <b>답할 수 있는 자료를 찾지 못했습니다.</b>
          <p style={{ margin: '6px 0 0', fontSize: 14, color: 'var(--muted)' }}>사유: {refusal}</p>
        </section>
      )}

      {reply && (
        <section style={{ marginTop: 24, padding: 16, border: '1px solid var(--line)', borderRadius: 10, whiteSpace: 'pre-wrap', fontSize: 15 }}>
          {reply}
          {busy === '작성' && <span style={{ color: 'var(--muted)' }}>▌</span>}
        </section>
      )}

      {verdict && <Judgement result={verdict} />}

      {reply && !busy && (
        <Feedback
          rated={rated}
          summary={fb}
          onRate={(liked) => {
            const v = verdict?.ok ? verdict.verdict : null
            setFb(feedbackSummary(saveFeedback({
              at: new Date().toISOString(),
              question: q,
              liked,
              judged: v ? { grounded: v.grounded, cited: v.cited, score: v.score } : null,
            })))
            setRated(true)
          }}
        />
      )}

      {openChunk && <EvidenceModal hit={openChunk} onClose={() => setOpenChunk(null)} />}

      {result && !refusal && <Result result={result} onOpen={setOpenChunk} />}

      <section style={{ marginTop: 32, padding: '14px 18px', border: '1px solid var(--line)', borderRadius: 10, background: 'var(--warn-bg)' }}>
        <h2 style={{ fontSize: 14, marginTop: 0, color: 'var(--warn)' }}>알아두실 점</h2>
        <ul style={{ paddingLeft: 18, margin: 0, fontSize: 14 }}>
          {LIMITS.map((t) => <li key={t} style={{ marginBottom: 4 }}>{t}</li>)}
          <li style={{ marginBottom: 4 }}>답변 아래 <b>근거로 쓴 자료</b>를 항상 함께 보여줍니다. 답이 자료와 맞는지 직접 확인하세요.</li>
        </ul>
      </section>
    </main>
  )
}

function Result({ result, onOpen }: { result: SearchResult; onOpen: (h: Hit) => void }) {
  if (result.verdict === 'refuse') return null

  return (
    <section style={{ marginTop: 24 }}>
      <h2 style={{ fontSize: 13, color: 'var(--muted)', fontWeight: 600 }}>근거로 쓴 자료</h2>
      {result.verdict === 'weak' && (
        <p style={{ display: 'inline-block', margin: '0 0 10px', padding: '3px 10px', borderRadius: 999, fontSize: 12, background: 'var(--warn-bg)', color: 'var(--warn)' }}>
          약한 근거 — {result.reason}
        </p>
      )}
      <ol style={{ paddingLeft: 20, margin: 0 }}>
        {result.hits.map((h) => (
          <li key={h.id} style={{ marginBottom: 14 }}>
            <div
              onClick={() => onOpen(h)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(h) } }}
              style={{ cursor: 'pointer' }}
              title="눌러서 원문과 출처 보기"
            >
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
              <Chip official={h.official}>{h.source_title}</Chip>
              {/* 어느 검색기로 들어왔는지 보여준다. 판단의 경로를 감추지 않는다 */}
              <span
                title={h.method === 'vector' ? '의미가 가까워서 뽑힘 (코사인)' : '질문의 낱말이 그대로 들어 있어서 뽑힘 (BM25)'}
                style={{ fontSize: 11, padding: '2px 7px', borderRadius: 999, border: '1px solid var(--line)', color: 'var(--muted)' }}
              >
                {h.method} {h.score.toFixed(3)}
              </span>
              <span style={{ fontSize: 11, color: 'var(--muted)' }}>{h.id} · {h.category} · {h.exam_mode}</span>
            </div>
            <div style={{ fontSize: 14 }}>{h.text.length > 140 ? h.text.slice(0, 140) + '…' : h.text}</div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

function Chip({ official, children }: { official: boolean; children: React.ReactNode }) {
  return (
    <span
      style={{
        fontSize: 11,
        padding: '2px 8px',
        borderRadius: 999,
        color: official ? 'var(--official)' : 'var(--unofficial)',
        background: official ? 'var(--official-bg)' : 'var(--unofficial-bg)',
      }}
    >
      {children}
      {!official && ' · 비공식'}
    </span>
  )
}

function Judgement({ result }: { result: JudgeResult }) {
  if (!result.ok) {
    return (
      <p style={{ marginTop: 12, fontSize: 12, color: 'var(--muted)' }}>
        판정 실패 — 답변과 근거는 그대로입니다. ({result.error})
      </p>
    )
  }
  const v = result.verdict
  const badge = (label: string, on: boolean, computed = false) => (
    <span
      key={label}
      title={computed ? '코드로 센 값입니다 (모델 판정 아님)' : '모델이 판정한 값입니다'}
      style={{
        fontSize: 11, padding: '2px 8px', borderRadius: 999,
        border: computed ? '1px solid var(--official)' : '1px solid var(--line)',
        color: on ? 'var(--fg)' : 'var(--muted)',
        opacity: on ? 1 : 0.55,
      }}
    >
      {label} {on ? '✓' : '✗'}
    </span>
  )

  return (
    <section style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
        <b style={{ fontSize: 12 }}>{v.score}점</b>
        {badge('근거성', v.grounded)}
        {badge('환각없음', v.noHalluc)}
        {badge('출처표기', v.cited, true)}
        {v.refusal && badge('정당한거부', true)}
      </div>
      {v.comment && <p style={{ margin: '6px 0 0', fontSize: 13, color: 'var(--muted)' }}>{v.comment}</p>}
      <p style={{ margin: '6px 0 0', fontSize: 11, color: 'var(--muted)' }}>
        ⚠️ 답변을 쓴 모델이 자기 답을 판정합니다. 독립 심사가 아니므로 신뢰의 근거로 쓰지 마세요.
        테두리가 있는 배지는 코드로 센 값입니다.
      </p>
    </section>
  )
}

/**
 * 근거 모달 — 칩을 누르면 원문 청크 전체와 출처 링크를 연다. (PRD v3 §10)
 *
 * 목록에서는 청크를 140자로 잘라 보여주므로, 잘린 부분을 확인할 길이 필요하다.
 * "근거를 공개한다"는 주장은 원문으로 돌아갈 수 있을 때만 성립한다.
 */
function EvidenceModal({ hit, onClose }: { hit: Hit; onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      role="presentation"
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, zIndex: 10,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="근거 원문"
        style={{
          background: 'var(--bg)', color: 'var(--fg)', border: '1px solid var(--line)',
          borderRadius: 12, padding: 20, maxWidth: 620, width: '100%', maxHeight: '80vh', overflowY: 'auto',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
          <div>
            <Chip official={hit.official}>{hit.source_title}</Chip>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>
              {hit.id} · {hit.category} · {hit.exam_mode} · {hit.method} {hit.score.toFixed(3)}
            </div>
          </div>
          <button onClick={onClose} aria-label="닫기" style={{ border: '1px solid var(--line)', background: 'transparent', color: 'inherit', borderRadius: 8, padding: '4px 10px', cursor: 'pointer' }}>
            닫기
          </button>
        </div>

        <p style={{ fontSize: 15, lineHeight: 1.7, marginTop: 16, whiteSpace: 'pre-wrap' }}>{hit.text}</p>

        {hit.source_url ? (
          <p style={{ marginTop: 16, fontSize: 13 }}>
            <a href={hit.source_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--official)' }}>
              원문 보기 ↗
            </a>
            <span style={{ color: 'var(--muted)', marginLeft: 8, wordBreak: 'break-all' }}>{hit.source_url}</span>
          </p>
        ) : (
          <p style={{ marginTop: 16, fontSize: 13, color: 'var(--muted)' }}>
            자체 작성 자료라 외부 출처 링크가 없습니다. 공식 자료가 아닙니다.
          </p>
        )}
      </div>
    </div>
  )
}

/**
 * 사람 피드백. (PRD v3 §11)
 *
 * 자동 판정과 **나란히** 읽으라고 불일치 건수를 함께 보여준다.
 * 생성 문장이 자연스럽다는 사실과 출처를 가진 답이라는 사실은 다르고,
 * 그 둘이 갈리는 지점이 곧 다시 읽어야 할 사례다.
 */
function Feedback({
  rated, summary, onRate,
}: {
  rated: boolean
  summary: { total: number; liked: number; disliked: number; mismatch: number }
  onRate: (liked: boolean) => void
}) {
  return (
    <section style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
      {rated ? (
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>평가 고맙습니다.</span>
      ) : (
        <>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>이 답변이 도움이 됐나요?</span>
          <button onClick={() => onRate(true)} style={{ padding: '4px 12px', borderRadius: 999, border: '1px solid var(--line)', background: 'transparent', cursor: 'pointer' }}>
            👍
          </button>
          <button onClick={() => onRate(false)} style={{ padding: '4px 12px', borderRadius: 999, border: '1px solid var(--line)', background: 'transparent', cursor: 'pointer' }}>
            👎
          </button>
        </>
      )}

      {summary.total > 0 && (
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>
          이 브라우저에 쌓인 평가 {summary.total}건 (👍 {summary.liked} · 👎 {summary.disliked})
          {summary.mismatch > 0 && (
            <b style={{ color: 'var(--warn)' }}> · 자동 판정과 어긋남 {summary.mismatch}건</b>
          )}
        </span>
      )}
    </section>
  )
}
