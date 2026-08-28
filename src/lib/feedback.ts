// 사람 피드백. (PRD v3 §11)
//
// 자동 판정과 **나란히** 읽을 수 있어야 한다. 그래서 피드백만 저장하지 않고
// 그때의 판정 결과(grounded / score)를 함께 저장한다.
// 둘이 어긋난 사례가 판정 기준이 놓친 것을 가리키기 때문이다.
//
// 브라우저 localStorage 에만 남는다. 서버로 보내지 않는다.

const KEY = 'topik-chatbot.feedback.v1'

export type Feedback = {
  at: string
  question: string
  /** 사람 판정 */
  liked: boolean
  /** 그때의 자동 판정 (없을 수도 있다 — 판정이 실패하면 null) */
  judged: { grounded: boolean; cited: boolean; score: number } | null
}

export function load(): Feedback[] {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Feedback[]) : []
  } catch {
    return [] // 시크릿 창 등에서 실패할 수 있다. 비어 있는 것으로 본다
  }
}

export function save(f: Feedback): Feedback[] {
  const all = [...load(), f]
  try {
    localStorage.setItem(KEY, JSON.stringify(all.slice(-200)))
  } catch {
    // 저장에 실패해도 화면은 계속 동작해야 한다
  }
  return all
}

export function clear(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* 무시 */
  }
}

/**
 * 사람과 자동 판정이 어긋난 건수.
 *
 * - 사람은 좋다는데 판정은 근거 없음 → 판정 기준이 놓친 유용성
 * - 사람은 나쁘다는데 판정은 근거 있음 → 근거는 맞지만 답이 도움이 안 된 경우
 *
 * 어느 쪽이든 **다시 읽어야 할 사례**다. 숫자만 세고 결론은 사람이 낸다.
 */
export function summary(all: Feedback[] = load()) {
  const judged = all.filter((f) => f.judged)
  const mismatch = judged.filter((f) => f.liked !== f.judged!.grounded).length
  return {
    total: all.length,
    liked: all.filter((f) => f.liked).length,
    disliked: all.filter((f) => !f.liked).length,
    mismatch,
  }
}
