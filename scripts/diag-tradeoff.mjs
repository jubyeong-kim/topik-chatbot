// 게이트 맞교환(22/25 → 20/25)의 원인을 찾는다.
//
// PRD 에 적어 둔 가정: "게이트가 받는 상위 4개의 구성이 바뀌어서".
// 그 가정을 검증하려면 먼저 **무엇이 실제로 달라졌는지** 를 LLM 없이 봐야 한다.
//
// 25문항 각각에 대해 index_text 적용 전/후의 GATE_K 개 청크 집합을 비교한다.
// 집합이 그대로인데 판정이 바뀌었다면 원인은 청크 구성이 아니라 모델의 비결정성이다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { GATE_K } from '../src/lib/ollama.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const after = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const before = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore-before.json'), 'utf8'))
const evalSet = JSON.parse(readFileSync(resolve(ROOT, 'eval/questions.json'), 'utf8'))

const UNANSWERABLE = [
  '54번에서 맞춤법을 하나 틀리면 몇 점이 깎이나요?',
  '53번 채점자는 몇 명인가요?',
  '쓰기 답안을 연필로 쓰면 감점인가요?',
  '54번에서 사자성어를 쓰면 가산점이 있나요?',
  '글자 수를 못 채우면 어떻게 되나요?',
]
const QUESTIONS = [...evalSet.questions.map((q) => q.q), ...UNANSWERABLE]

const embed = await pipeline('feature-extraction', after.model, { dtype: after.dtype })
const idxA = buildBm25Index(after.chunks)
const idxB = buildBm25Index(before.chunks)

let same = 0, changed = 0
const rows = []

for (const q of QUESTIONS) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const v = Array.from(out.data)

  const rA = search(after.chunks, idxA, q, v)
  const rB = search(before.chunks, idxB, q, v)
  if (rA.verdict === 'refuse') continue // 1층에서 걸린 것은 게이트를 안 탄다

  const a = rA.hits.slice(0, GATE_K).map((h) => h.id)
  const b = rB.hits.slice(0, GATE_K).map((h) => h.id)
  const identical = a.join() === b.join()
  identical ? same++ : changed++

  rows.push({ q, a, b, identical })
  console.log(
    `${identical ? '=' : '≠'} ${q.slice(0, 34).padEnd(36)}` +
      (identical ? `동일  ${a.join(' ')}` : `\n    전: ${b.join(' ')}\n    후: ${a.join(' ')}`),
  )
}

console.log(`\n게이트를 타는 ${rows.length}문항 중 · 청크 집합 동일 ${same} · 달라짐 ${changed}`)
console.log(
  changed === 0
    ? '\n→ 청크 구성은 하나도 안 바뀌었다. 그렇다면 22→20 은 index_text 때문이 아니다.\n  게이트의 비결정성을 의심해야 한다.'
    : `\n→ ${changed}문항의 청크가 바뀌었다. 판정이 뒤집힌 문항이 이 안에 있는지 확인해야 한다.`,
)
