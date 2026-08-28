// 로컬 Ollama 연결. (PRD v3 §9)
//
// 브라우저에서 http://localhost:11434 로 직접 부른다. 서버는 없다.
// GitHub Pages 에서 열면 출처가 달라 CORS 로 막히므로 OLLAMA_ORIGINS 설정이 필요하다.
// 로컬(npm run dev)에서는 필요 없다.

import type { Hit } from './search'

export const OLLAMA = 'http://localhost:11434'
export const MODEL = 'qwen2.5:1.5b-instruct-q4_K_M'

// 게이트와 생성에 넣는 청크 수는 다르다. 한 곳에 모아 두 값이 어긋나지 않게 한다.
// (실제로 어긋난 적이 있다 — 앱은 8개, 평가 스크립트는 5개를 넣어 측정과 동작이 달랐다.)
//
// 지연은 전부 프롬프트를 읽는 시간이고, 속도는 설정과 무관하게 30~40 tok/s 로 일정하다.
// 즉 게이트 시간 ≈ 읽을 토큰 수 ÷ 35. 생성 토큰은 1~2개뿐이라 num_predict 는 레버가 아니다.
// (scripts/bench-gate2.mjs — 청크 8개 29초 / 4개 14초)
export const GATE_K = 4   // 🔴 임의 — 답이 있다면 상위권에 있다는 가정. §12 에서 검증
export const ANSWER_K = 8 // 🔴 임의 — 5개로 줄였더니 채점 기준표(코사인 6위)가 잘렸다

export type Health =
  | { ok: true; version: string; hasModel: boolean }
  | { ok: false; reason: string }

export async function health(base = OLLAMA): Promise<Health> {
  try {
    const [v, tags] = await Promise.all([
      fetch(`${base}/api/version`).then((r) => r.json()),
      fetch(`${base}/api/tags`).then((r) => r.json()),
    ])
    const hasModel = (tags.models ?? []).some((m: { name: string }) => m.name === MODEL)
    return { ok: true, version: v.version, hasModel }
  } catch {
    return {
      ok: false,
      reason:
        'Ollama 에 연결하지 못했습니다. 설치 후 실행 중인지 확인하세요. ' +
        '배포된 주소에서 열었다면 OLLAMA_ORIGINS 설정이 필요합니다.',
    }
  }
}

function evidence(hits: Hit[]): string {
  return hits.map((h, i) => `[${i + 1}] (${h.id}) ${h.text}`).join('\n')
}

// ── 2층: LLM 게이트 ──────────────────────────────────────────
// 범위 규칙(1층)은 어휘로 드러나는 범위 밖만 잡는다.
// "54번에서 맞춤법 하나 틀리면 몇 점 깎이나요?" 처럼 범위 안처럼 보이지만
// 자료에 답이 없는 질문은 여기서 걸러야 한다.

// 판정 기준은 PRD §1 문구를 그대로 옮긴다 — "공개 자료에 근거해 답한다".
// "자료에 그대로 적혀 있는가" 로 좁히면 해설 청크로 답할 수 있는 질문까지 거절된다.
// ⚠️ 출력 어휘를 바꾸지 말 것. 모델은 프롬프트에 쓰인 낱말을 그대로 따라 한다.
//    "answerable = true/false" 로 적었더니 모델이 그 문자열을 뱉어 파싱이 전부 깨졌다.
//    판정 기준은 PRD §1("공개 자료에 근거해 답한다")을 따르되, 출력은 YES/NO 로 고정한다.
const GATE_SYSTEM = `당신은 자료 검토자입니다. 답을 만들지 마십시오.
주어진 자료만 보고, 질문에 답할 근거가 자료 안에 있는지만 판단합니다.

YES: 자료에 적힌 내용만으로 질문에 답할 수 있다.
NO: 자료가 주제만 비슷할 뿐, 질문이 묻는 사실이 자료에 없다.

당신의 배경지식은 쓰지 마십시오. 자료에 없으면 NO 입니다.
YES 또는 NO 한 단어만 출력하십시오.`

export async function gate(
  question: string,
  hits: Hit[],
  opts: { base?: string; signal?: AbortSignal } = {},
): Promise<{ answerable: boolean; raw: string }> {
  const res = await fetch(`${opts.base ?? OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      model: MODEL,
      stream: false,
      options: { temperature: 0, num_predict: 5 },
      // ⚠️ JSON 스키마 강제(format)를 써 봤더니 정확도가 10/15 → 5/15 로 떨어졌다.
      //    1.5B 모델은 형식을 맞추는 데 용량을 쓰느라 판단이 나빠진다.
      //    형식은 자유 출력에 맡기고, 파싱을 튼튼하게 하는 쪽이 낫다.
      messages: [
        { role: 'system', content: GATE_SYSTEM },
        { role: 'user', content: `자료:\n${evidence(hits)}\n\n질문: ${question}\n\nYES 또는 NO:` },
      ],
    }),
  })
  if (!res.ok) throw new Error(`게이트 호출 실패 (${res.status})`)
  const json = await res.json()
  const raw = (json.message?.content ?? '').trim()

  // YES/NO 중 무엇이 먼저 나오는지로 읽는다. 둘 다 없으면 판정 불가로 남긴다 —
  // 조용히 거절로 처리하면 모델의 형식 실패가 "자료 없음" 으로 둔갑한다.
  const yes = raw.search(/yes|예\b/i)
  const no = raw.search(/no|아니/i)
  if (yes < 0 && no < 0) return { answerable: false, raw: `판정불가: ${raw}` }
  return { answerable: no < 0 || (yes >= 0 && yes < no), raw }
}

// ── 답변 생성 ────────────────────────────────────────────────

const ANSWER_SYSTEM = `당신은 TOPIK 쓰기 채점 안내봇입니다.
아래 규칙을 반드시 지키십시오.

1. 주어진 자료에 적힌 내용만으로 답합니다. 배경지식을 더하지 마십시오.
2. 자료에 없는 것은 "자료에 없습니다"라고 밝힙니다. 추측하지 마십시오.
3. PBT 와 IBT 는 문항 수와 배점이 다릅니다. 질문이 어느 쪽인지 밝히지 않았다면
   답변에 어느 방식 기준인지 함께 적으십시오.
4. 한국어로, 군더더기 없이 답하십시오. 같은 항목을 두 번 쓰지 마십시오.`
// ⚠️ 형식을 별도 템플릿 블록으로 강하게 지시했더니, 모델이 그 템플릿만 출력하고
//    답변 본문이 통째로 사라졌다. 괄호 안 설명("자료의 방식을 그대로")까지 그대로 베꼈다.
//    1.5B 는 프롬프트의 문장을 흉내 내는 경향이 강하다. 규칙은 목록 한 줄로 짧게 둔다.

/** 질문 시점의 서울 날짜. '올해', '지금' 같은 상대 표현의 해석 기준이 된다 (PRD §10) */
function todayKST(): string {
  return new Intl.DateTimeFormat('ko-KR', {
    timeZone: 'Asia/Seoul',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(new Date())
}

export async function* answer(
  question: string,
  hits: Hit[],
  opts: { base?: string; signal?: AbortSignal; weak?: boolean } = {},
): AsyncGenerator<string> {
  const caution = opts.weak
    ? '\n\n주의: 아래 자료는 질문과 연관이 약합니다. 자료에 분명히 적힌 것만 짧게 답하고, 나머지는 자료에 없다고 밝히십시오.'
    : ''

  const res = await fetch(`${opts.base ?? OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      model: MODEL,
      stream: true,
      options: { temperature: 0.2 },
      messages: [
        // 시간 컨텍스트는 자료 앞에 둔다 (PRD §10). 한 줄로 짧게 —
        // 문장을 길게 쓰면 모델이 그 문장을 답변에 베낀다.
        {
          role: 'system',
          content: `${ANSWER_SYSTEM}${caution}\n\n오늘 날짜는 ${todayKST()}입니다. '지금', '올해' 같은 표현은 이 날짜를 기준으로 해석하십시오.`,
        },
        // ⚠️ [ID] 표기를 프롬프트 맨 끝으로 옮겨 봤다 (§12 회차 H).
        //    인용률은 0/2 → 2/2 로 올랐지만, 한 답변이 80자로 쪼그라들어
        //    본문이 사라지고 "[TPK-024]" 만 남았다. 끝에 둔 지시가 답변을 삼킨다.
        //    → 되돌렸다. 답변 본문이 인용 표기보다 중요하다.
        //    추적 가능성은 화면의 '근거로 쓴 자료' 목록이 이미 제공한다.
        { role: 'user', content: `자료:\n${evidence(hits)}\n\n질문: ${question}` },
      ],
    }),
  })
  if (!res.ok || !res.body) throw new Error(`답변 생성 실패 (${res.status})`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      const json = JSON.parse(line)
      const piece = json.message?.content
      if (piece) yield piece
    }
  }
}
