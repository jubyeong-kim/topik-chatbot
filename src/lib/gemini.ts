// 선택 엔진 — Gemini API. (PRD v3 §9-B)
//
// ⚠️ 기본값이 아니다. 켜야만 동작한다.
//
// PRD §1 은 "외부 데이터 유출 없는 구조" 를 전제로 삼는다. 이 엔진을 켜면 그 전제가 깨진다 —
// 질문과 검색된 근거 청크가 Google 서버로 나간다. 그래서
//   1) 기본은 항상 로컬 Ollama 다
//   2) 켤 때 무엇이 나가는지 화면에 밝힌다
//   3) 켠 동안에는 화면에 계속 표시한다
//
// 🔑 API 키는 **사용자가 직접 입력**하고 그 브라우저에만 남는다.
//    저장소에도 번들에도 키가 들어가지 않는다 — 정적 사이트에서 키를 번들에 넣으면
//    누구나 읽을 수 있기 때문이다. 서버가 없으니 숨길 곳도 없다.

import type { Hit } from './search'

const BASE = 'https://generativelanguage.googleapis.com/v1beta'
export const GEMINI_MODEL = 'gemini-3.6-flash'
const KEY_STORE = 'topik-chatbot.gemini-key'

export function getKey(): string {
  try {
    return localStorage.getItem(KEY_STORE) ?? ''
  } catch {
    return ''
  }
}

export function setKey(k: string): void {
  try {
    k ? localStorage.setItem(KEY_STORE, k) : localStorage.removeItem(KEY_STORE)
  } catch {
    /* 시크릿 창 등에서 실패할 수 있다 */
  }
}

function evidence(hits: Hit[]): string {
  return hits.map((h, i) => `[${i + 1}] (${h.id}) ${h.text}`).join('\n')
}

async function* sse(res: Response): AsyncGenerator<string> {
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => '')
    throw new Error(`Gemini 호출 실패 (${res.status}) ${body.slice(0, 160)}`)
  }
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (!payload || payload === '[DONE]') continue
      try {
        const text = JSON.parse(payload)?.candidates?.[0]?.content?.parts?.[0]?.text
        if (text) yield text
      } catch {
        // 잘린 조각은 건너뛴다. 다음 청크에서 이어진다
      }
    }
  }
}

/** 게이트 — 로컬과 같은 질문을 던져 같은 기준으로 비교할 수 있게 한다 */
export async function gate(
  question: string,
  hits: Hit[],
  system: string,
  opts: { key?: string; signal?: AbortSignal } = {},
): Promise<{ answerable: boolean; raw: string }> {
  const key = opts.key ?? getKey()
  const res = await fetch(`${BASE}/models/${GEMINI_MODEL}:generateContent?key=${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: 'user', parts: [{ text: `자료:\n${evidence(hits)}\n\n질문: ${question}\n\nYES 또는 NO:` }] }],
      generationConfig: { temperature: 0, maxOutputTokens: 500 },
    }),
  })
  // 429 는 무료 등급의 분당 요청 한도다. 실패로 끝내지 않고 기다렸다 다시 부른다 —
  // 25문항을 연속으로 돌리면 반드시 만난다.
  if (res.status === 429) {
    const wait = Number(res.headers.get('retry-after') ?? 20) * 1000
    await new Promise((r) => setTimeout(r, wait))
    return gate(question, hits, system, opts)
  }
  if (!res.ok) throw new Error(`Gemini 게이트 실패 (${res.status})`)
  const raw = ((await res.json())?.candidates?.[0]?.content?.parts?.[0]?.text ?? '').trim()
  const yes = raw.search(/yes/i)
  const no = raw.search(/no/i)
  if (yes < 0 && no < 0) return { answerable: false, raw: `판정불가: ${raw}` }
  return { answerable: no < 0 || (yes >= 0 && yes < no), raw }
}

export async function* answer(
  question: string,
  hits: Hit[],
  system: string,
  opts: { key?: string; signal?: AbortSignal } = {},
): AsyncGenerator<string> {
  const key = opts.key ?? getKey()
  const res = await fetch(
    `${BASE}/models/${GEMINI_MODEL}:streamGenerateContent?alt=sse&key=${encodeURIComponent(key)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: opts.signal,
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: 'user', parts: [{ text: `자료:\n${evidence(hits)}\n\n질문: ${question}` }] }],
        generationConfig: { temperature: 0.2 },
      }),
    },
  )
  yield* sse(res)
}

/** 키가 실제로 동작하는지 확인한다. 저장 전에 부른다 */
export async function check(key: string): Promise<{ ok: true } | { ok: false; reason: string }> {
  try {
    const res = await fetch(`${BASE}/models/${GEMINI_MODEL}?key=${encodeURIComponent(key)}`)
    if (res.ok) return { ok: true }
    if (res.status === 400 || res.status === 403) return { ok: false, reason: '키가 올바르지 않거나 권한이 없습니다.' }
    return { ok: false, reason: `확인 실패 (${res.status})` }
  } catch {
    return { ok: false, reason: '네트워크 오류로 키를 확인하지 못했습니다.' }
  }
}
