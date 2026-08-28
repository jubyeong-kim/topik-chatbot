// 2층 LLM 게이트를 측정한다. 이게 §1 "자료 밖은 무조건 거절" 의 마지막 방어선이다.
//
// 세 묶음을 넣는다.
//   봉인 자료 안 15개  → 대부분 YES 여야 한다 (과잉거절이면 봇이 쓸모없어진다)
//   봉인 자료 밖 5개   → 1층에서 이미 잡히지만, 게이트 단독 성능도 본다
//   진단 4개           → 범위 안처럼 보이나 자료에 답이 없다. 여기서 NO 가 나와야 한다

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { gate, health, GATE_K } from '../src/lib/ollama.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const evalSet = JSON.parse(readFileSync(resolve(ROOT, 'eval/questions.json'), 'utf8'))

const h = await health()
if (!h.ok) { console.error(h.reason); process.exit(1) }
console.log(`Ollama ${h.version} · 모델 ${h.hasModel ? '있음' : '없음'}\n`)

// 범위 안처럼 보이지만 자료에 답이 없는 질문 (봉인 세트 아님 — 진단용)
const UNANSWERABLE = [
  '54번에서 맞춤법을 하나 틀리면 몇 점이 깎이나요?',
  '53번 채점자는 몇 명인가요?',
  '쓰기 답안을 연필로 쓰면 감점인가요?',
  '54번에서 사자성어를 쓰면 가산점이 있나요?',
  '글자 수를 못 채우면 어떻게 되나요?', // diag-rank 에서 코퍼스에 답이 없다고 확인된 것
]

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

async function ask(q) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const r = search(store.chunks, index, q, Array.from(out.data))
  if (r.verdict === 'refuse') return { layer: 1, answerable: false, raw: r.reason }
  const g = await gate(q, r.hits.slice(0, GATE_K))
  return { layer: 2, ...g }
}

async function run(title, questions, want) {
  console.log(`\n[${title}] 기대: ${want ? 'YES(답변)' : 'NO(거절)'}`)
  let ok = 0
  for (const q of questions) {
    const r = await ask(q)
    const pass = r.answerable === want
    if (pass) ok++
    console.log(`  ${pass ? '✓' : '✗'} ${r.answerable ? 'YES' : 'NO '} (${r.layer}층${r.layer === 2 ? ` "${r.raw}"` : ''})  ${q}`)
  }
  console.log(`  → ${ok}/${questions.length}`)
  return [ok, questions.length]
}

const inC = evalSet.questions.filter((q) => q.expect === 'in_corpus').map((q) => q.q)
const outC = evalSet.questions.filter((q) => q.expect === 'refuse').map((q) => q.q)

const a = await run('봉인 · 자료 안', inC, true)
const b = await run('봉인 · 자료 밖', outC, false)
const c = await run('진단 · 범위 안이나 자료 없음', UNANSWERABLE, false)

console.log(`\n합계 ${a[0] + b[0] + c[0]}/${a[1] + b[1] + c[1]}`)
console.log(`  과잉거절 ${a[1] - a[0]}건 · 과소거절 ${(b[1] - b[0]) + (c[1] - c[0])}건`)
