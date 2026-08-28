// 검색과 거절 판정. (PRD v3 §10)
//
// ⚠️ 실측으로 뒤집힌 설계다. 코사인 임계값 거절은 폐기했다.
//
// 봉인 질문 20개로 측정한 결과, 자료 안 질문의 top1 코사인은 0.838~0.900,
// 자료 밖 질문은 0.850~0.875 로 **완전히 겹쳤다.** top1-평균, z-점수, 상위권 낙폭도
// 전부 겹쳤다. 자료 밖 질문도 다 TOPIK 이야기라, 임베딩이 재는 "주제 근접성"으로는
// "이 청크가 이 질문의 답이 되는가"를 가릴 수 없다. (scripts/diag-threshold.mjs)
//
// 그래서 거절을 3층으로 바꿨다.
//   1층 범위 규칙  — PRD §1·§3 의 범위 정의에서 끌어낸 결정적 필터. 설명 가능
//   2층 LLM 게이트 — "이 자료로 답할 수 있는가" 를 모델에 묻는다 (llmGate 주입)
//   3층 약한 근거  — 어휘 일치가 없으면 배지를 단다. 거절은 아니고 표시만
//
// BM25 합성 점수는 순위에만 쓴다. 거절 판정에는 쓰지 않는다 —
// 정규화가 질의 내 상대값이라 자료가 없어도 최상위가 1.0 이 되기 때문이다.

export type Chunk = {
  id: string
  text: string
  category: string
  source_title: string
  source_url: string | null
  official: boolean
  exam_mode: 'PBT' | 'IBT' | '공통'
  vector: number[]
}

/** 이 후보가 어느 검색기로 들어왔는지. 화면에 그대로 보여준다 */
export type Method = 'vector' | 'bm25'

export type Hit = Chunk & {
  method: Method
  /** method 기준 점수. vector 면 코사인, bm25 면 최고점 대비 0~1 */
  score: number
  cosine: number
  bm25: number
  salient: boolean
}

export type SearchResult = {
  verdict: 'answer' | 'weak' | 'refuse'
  /** 거절·약한 근거의 사유. 사용자에게 그대로 보여줄 수 있어야 한다 */
  reason: string | null
  hits: Hit[]
  /** 참고용. 이 값으로 거절을 판정하지 않는다 — 자료 안/밖이 겹친다 */
  topCosine: number
  salientTerms: string[]
}

export const DEFAULTS = {
  /**
   * 약한 근거 경계. 🔴 임의 — §12 실험으로 정한다.
   * ⚠️ 이 값으로 결과를 버리지 않는다. 근거는 그대로 넘기고 말투만 보수적으로 바꾼다.
   * 우리 코퍼스의 코사인은 0.78~0.90 범위다. 다른 코퍼스의 임계값(예: 0.55)을 그대로 옮기면
   * 영원히 걸리지 않거나 항상 걸린다.
   */
  cosineThreshold: 0.82,
  /** 코사인으로 먼저 뽑는 개수 */
  vectorK: 10,
  /** 그 뒤에 BM25 로 덧붙이는 개수 (코사인 후보와 중복 제외) */
  bm25K: 5,
  /** BM25 길이 정규화. 높을수록 긴 청크에 불리하다 (§12 실험 대상) */
  bm25B: 0.75,
}

// ── 토크나이저 ────────────────────────────────────────────────
// 한국어는 조사가 붙어 다녀 어절 단위로 자르면 "54번은" 과 "54번" 이 다른 토큰이 된다.
// 한글은 문자 바이그램으로, 숫자·영문은 통째로 자른다 (CJK 검색의 표준적인 처리).
export function tokenize(s: string): string[] {
  const out: string[] = []
  const runs = s.toLowerCase().match(/[0-9]+|[a-z]+|[가-힣]+/g) ?? []
  for (const run of runs) {
    if (/^[가-힣]+$/.test(run)) {
      if (run.length === 1) out.push(run)
      for (let i = 0; i + 1 < run.length; i++) out.push(run.slice(i, i + 2))
    } else {
      out.push(run)
    }
  }
  return out
}

/**
 * 질문에서 "이것이 있으면 그 청크가 정답일 가능성이 높은" 어구를 뽑는다.
 * 문항번호·점수·분량 같은 것들로, 임베딩이 약하고 어휘 일치가 강한 유형이다.
 */
export function salientTerms(q: string): string[] {
  const terms = new Set<string>()
  for (const m of q.matchAll(/\d+\s*(번|점|급|자|분|문항)/g)) terms.add(m[0].replace(/\s+/g, ''))
  for (const m of q.matchAll(/\b(PBT|IBT)\b/gi)) terms.add(m[0].toUpperCase())
  for (const m of q.matchAll(/'([^']{2,20})'|“([^”]{2,20})”/g)) terms.add((m[1] ?? m[2]).trim())
  return [...terms]
}

// ── 1층: 범위 규칙 ────────────────────────────────────────────
// ⚠️ 이 목록은 PRD §1(범위 = TOPIK 쓰기 채점)과 §3(제외 목록)에서 끌어냈다.
//    봉인 질문을 보고 맞춘 것이 아니다. 봉인 세트에 맞춰 이 목록을 고치면
//    측정이 무의미해지므로, 어긋나는 사례가 나와도 여기를 고치지 말고 기록한다.
const OUT_OF_SCOPE: { reason: string; terms: string[] }[] = [
  { reason: '쓰기 외 영역', terms: ['듣기', '읽기', '말하기'] },
  { reason: '시험 운영', terms: ['응시료', '접수', '시험장', '성적표', '시험 일정', '환불', '준비물', '재응시'] },
  // '가산점' 은 뺐다 — "취업 가산점"(범위 밖)과 "54번 가산점"(범위 안)을 못 가른다.
  // 진단에서 실제로 오탐이 났다 (scripts/probe-unanswerable.mjs).
  { reason: '진로·자격', terms: ['비자', '유학', '취업', '대학원', '입학', '이민'] },
  { reason: '교수법·학습법', terms: ['가르치', '교수법', '지도법', '공부법', '학습법', '어떻게 공부'] },
  { reason: '개인 답안 첨삭', terms: ['첨삭', '고쳐 주', '고쳐주', '봐 주세요', '봐주세요', '채점해 주', '채점해주'] },
]

/** 범위 밖이면 사유를, 아니면 null 을 돌려준다. */
export function outOfScope(q: string): string | null {
  for (const g of OUT_OF_SCOPE) {
    if (g.terms.some((t) => q.includes(t))) return g.reason
  }
  return null
}

// ── 점수 ─────────────────────────────────────────────────────
export function cosine(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s // 저장 시 정규화되어 있으므로 내적이 곧 코사인
}

type Bm25Index = { df: Map<string, number>; docLen: number[]; avgLen: number; N: number; docs: string[][] }

export function buildBm25Index(chunks: Chunk[]): Bm25Index {
  const docs = chunks.map((c) => tokenize(c.text))
  const df = new Map<string, number>()
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1)
  const docLen = docs.map((d) => d.length)
  return { df, docLen, avgLen: docLen.reduce((a, b) => a + b, 0) / docs.length, N: docs.length, docs }
}

export function bm25Scores(index: Bm25Index, query: string, k1 = 1.5, b = 0.75): number[] {
  const q = tokenize(query)
  return index.docs.map((doc, i) => {
    const tf = new Map<string, number>()
    for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1)
    let score = 0
    for (const t of q) {
      const f = tf.get(t)
      if (!f) continue
      const idf = Math.log(1 + (index.N - (index.df.get(t) ?? 0) + 0.5) / ((index.df.get(t) ?? 0) + 0.5))
      score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * index.docLen[i]) / index.avgLen)))
    }
    return score
  })
}

// ── 검색 ─────────────────────────────────────────────────────
export function search(
  chunks: Chunk[],
  index: Bm25Index,
  queryText: string,
  queryVector: number[],
  opts: Partial<typeof DEFAULTS> = {},
): SearchResult {
  const { cosineThreshold, vectorK, bm25K, bm25B } = { ...DEFAULTS, ...opts }

  const cosines = chunks.map((c) => cosine(c.vector, queryVector))
  const bm25 = bm25Scores(index, queryText, 1.5, bm25B)
  const maxBm = Math.max(...bm25, 1e-9)
  const terms = salientTerms(queryText)
  const topCosine = Math.max(...cosines)

  const base = (c: Chunk, i: number) => ({
    ...c,
    cosine: cosines[i],
    bm25: bm25[i],
    salient: terms.length > 0 && terms.every((t) => c.text.includes(t)),
  })

  // 두 검색기를 점수로 섞지 않는다. 각자 뽑게 하고 합친다.
  //
  // 가중합(0.7*cos + 0.3*bm25)을 쓰면 짧은 청크가 낱말 하나로 점수를 벌어
  // 긴 근거 청크를 밀어낸다. 실측에서 top-1 적중이 6/15 로, 코사인 단독 10/15 보다 나빴다.
  // (scripts/diag-rank.mjs) 합집합은 그 오염이 구조적으로 생기지 않는다.
  const order = chunks.map((_, i) => i)

  const vectorPicks = [...order].sort((a, b) => cosines[b] - cosines[a]).slice(0, vectorK)
  const taken = new Set(vectorPicks)

  const bm25Picks = [...order]
    .filter((i) => !taken.has(i) && bm25[i] > 0)
    .sort((a, b) => bm25[b] - bm25[a])
    .slice(0, bm25K)

  const top: Hit[] = [
    ...vectorPicks.map((i) => ({ ...base(chunks[i], i), method: 'vector' as const, score: cosines[i] })),
    ...bm25Picks.map((i) => ({ ...base(chunks[i], i), method: 'bm25' as const, score: bm25[i] / maxBm })),
  ]

  // 1층 — 범위 규칙. 검색조차 하지 않고 거절한다.
  const scope = outOfScope(queryText)
  if (scope) {
    return { verdict: 'refuse', reason: `범위 밖 (${scope})`, hits: [], topCosine, salientTerms: terms }
  }

  // 2층(LLM 게이트)은 호출부에서 이 결과를 받아 수행한다. 여기서는 후보를 넘긴다.
  // 3층 — 어휘 일치도 없고 유사도도 낮으면 '약한 근거' 로 표시만 한다. 거절이 아니다.
  const weak = !top.some((h) => h.salient) && topCosine < cosineThreshold

  return {
    verdict: weak ? 'weak' : 'answer',
    reason: weak ? '어휘 일치 없음' : null,
    hits: top,
    topCosine,
    salientTerms: terms,
  }
}
