// 답변 뒤에 두는 평가자. (PRD v3 §11 · 학습 목표 4)
//
// ⚠️ 이것은 독립 심사가 아니다. 답변을 쓴 모델과 판정하는 모델이 같은 1.5B 다.
//    자기가 쓴 답을 자기가 채점하는 구조이므로 판정을 신뢰의 근거로 쓸 수 없다.
//    화면에도 이 한계를 적는다. 숨기면 배지가 신뢰의 도장처럼 읽힌다.
//
// 우리 구조에서 거절은 생성 **전에** 일어난다(1층 범위 규칙, 2층 게이트).
// 따라서 모델이 "자료에 없습니다"라고 쓰는 일 자체가 드물고,
// refusal 필드는 답변 안에서 부분적으로 거절했는지를 본다.

import type { Hit } from './search'
import { MODEL, OLLAMA } from './ollama.ts'

export type Verdict = {
  grounded: boolean
  noHalluc: boolean
  /** 코드로 센 값. 모델 판정이 아니다 */
  cited: boolean
  citedIds: string[]
  refusal: boolean
  /** grounded 40 + noHalluc 40 + cited 20 으로 계산한 값 */
  score: number
  comment: string
}

export type JudgeResult = { ok: true; verdict: Verdict; raw: string } | { ok: false; error: string; raw: string }

// ⚠️ cited 와 score 는 모델에게 묻지 않는다. 코드로 센다.
//
// 실측: 1.5B 판정기에게 cited 를 물었더니 답변에 [ID] 표기가 **하나도 없는데**
// 두 건 모두 cited=true 로 답했다. 기계적으로 확인 가능한 사실에서 틀린 것이다.
// 게다가 noHalluc=false 라면서 score=85 를 주는 등 필드 간 모순도 냈다.
// 셀 수 있는 것을 모델에게 묻는 것은 판정기를 약하게 만들 뿐이다.
//
// 모델에게는 세는 것으로 대신할 수 없는 것만 맡긴다 — grounded / noHalluc / refusal / comment.
const JUDGE_SYSTEM = `당신은 RAG 챗봇 답변의 평가자입니다. 답을 다시 쓰지 마십시오.
질문에 답하지 말고, 답변을 평가만 하십시오.
[질문], [근거자료], [답변]을 읽고 아래 기준으로 JSON만 출력합니다.

grounded: 답변 내용이 근거자료에서 나왔는가 (true/false)
allFromSource: 답변에 나온 사실이 모두 자료 안에 있는가 (true/false)
refusal: 근거에 답이 없어서 '없다'고 답한 경우 true, 그 외 false
comment: 답변을 평가하는 한두 문장 (한국어). 질문에 대한 답이 아니라 답변에 대한 평가입니다.

출력 형식: {"grounded":bool,"allFromSource":bool,"refusal":bool,"comment":"..."}
JSON 외 텍스트 금지.`

export async function judge(
  question: string,
  hits: Hit[],
  reply: string,
  opts: { base?: string; signal?: AbortSignal } = {},
): Promise<JudgeResult> {
  const evidence = hits.map((h) => `[${h.id} | ${h.category}] ${h.text}`).join('\n')

  let raw = ''
  try {
    const res = await fetch(`${opts.base ?? OLLAMA}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: opts.signal,
      body: JSON.stringify({
        model: MODEL,
        stream: false,
        format: 'json',
        options: { temperature: 0 },
        messages: [
          { role: 'system', content: JUDGE_SYSTEM },
          { role: 'user', content: `[질문]\n${question}\n\n[근거자료]\n${evidence}\n\n[답변]\n${reply}` },
        ],
      }),
    })
    if (!res.ok) return { ok: false, error: `판정 호출 실패 (${res.status})`, raw }
    raw = ((await res.json()).message?.content ?? '').trim()

    const v = JSON.parse(raw)

    // cited 는 코드로 센다. 모델에게 묻지 않는다.
    const cited = citedIds(reply, hits)
    const grounded = v.grounded === true
    // 이중부정('지어내지 않았는가')이면 1.5B 가 방향을 놓쳐 늘 false 를 냈다(4/4).
    // 긍정형('모두 자료 안에 있는가')으로 바꿔 본다 — §12 회차 I
    const noHalluc = v.allFromSource === true

    // score 도 모델에게 묻지 않는다. 세 축을 그대로 합산해 계산이 보이게 한다.
    const score = (grounded ? 40 : 0) + (noHalluc ? 40 : 0) + (cited.length > 0 ? 20 : 0)

    return {
      ok: true,
      raw,
      verdict: {
        grounded,
        noHalluc,
        cited: cited.length > 0,
        citedIds: cited,
        refusal: v.refusal === true,
        score,
        comment: typeof v.comment === 'string' ? v.comment : '',
      },
    }
  } catch (e) {
    // 판정이 실패해도 답변과 출처는 그대로 둔다. 배지만 실패로 남긴다.
    return { ok: false, error: e instanceof Error ? e.message : String(e), raw }
  }
}

/** 답변 안의 [TPK-024] 같은 표기를 직접 센다. cited 는 모델에게 묻지 않고도 확인할 수 있다. */
export function citedIds(reply: string, hits: Hit[]): string[] {
  const ids = new Set(hits.map((h) => h.id))
  return [...new Set(Array.from(reply.matchAll(/\b([A-Z]{3}-\d{3})\b/g), (m) => m[1]))].filter((id) => ids.has(id))
}
