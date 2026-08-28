// 맞교환의 원인을 확정한다. 두 가지를 같은 실행에서 본다.
//
//   1) 같은 질문을 전/후 벡터스토어로 각각 게이트에 태워 **어느 문항이 뒤집혔는지**
//   2) 후 벡터스토어로 **한 번 더** 태워 게이트가 결정적인지
//
// 2번이 중요하다. 같은 입력에 다른 판정이 나오면 22 vs 20 은 변경 효과가 아니라 잡음이다.
// temperature 0 이라도 결정적이라는 보장은 없다 — 확인해야 안다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { gate, GATE_K } from '../src/lib/ollama.ts'

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
const ITEMS = [
  ...evalSet.questions.map((x) => ({ q: x.q, want: x.expect === 'in_corpus' })),
  ...UNANSWERABLE.map((q) => ({ q, want: false })),
]

const embed = await pipeline('feature-extraction', after.model, { dtype: after.dtype })
const idxA = buildBm25Index(after.chunks)
const idxB = buildBm25Index(before.chunks)

let flips = 0, unstable = 0, scoreB = 0, scoreA = 0
console.log('       전   후1  후2   기대   질문')
console.log('-'.repeat(78))

for (const { q, want } of ITEMS) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const v = Array.from(out.data)
  const rA = search(after.chunks, idxA, q, v)
  const rB = search(before.chunks, idxB, q, v)

  if (rA.verdict === 'refuse') { // 1층에서 끝 — 게이트를 안 탄다
    scoreB += want ? 0 : 1
    scoreA += want ? 0 : 1
    continue
  }

  const gB = (await gate(q, rB.hits.slice(0, GATE_K))).answerable
  const gA1 = (await gate(q, rA.hits.slice(0, GATE_K))).answerable
  const gA2 = (await gate(q, rA.hits.slice(0, GATE_K))).answerable

  if (gB === want) scoreB++
  if (gA1 === want) scoreA++
  const flipped = gB !== gA1
  const shaky = gA1 !== gA2
  if (flipped) flips++
  if (shaky) unstable++

  const y = (b) => (b ? 'YES' : 'NO ')
  console.log(
    `${flipped ? '뒤집' : '    '} ${shaky ? '흔들' : '    '} ${y(gB)} ${y(gA1)} ${y(gA2)}  ${y(want)}  ${q.slice(0, 30)}`,
  )
}

console.log(`\n전 벡터스토어 ${scoreB}/${ITEMS.length} · 후 벡터스토어 ${scoreA}/${ITEMS.length}`)
console.log(`판정이 뒤집힌 문항 ${flips}개 · 같은 입력에 판정이 흔들린 문항 ${unstable}개`)
console.log(
  unstable > 0
    ? `\n→ 게이트가 결정적이지 않다. 같은 입력에 ${unstable}개가 다른 답을 냈다.\n  그렇다면 22 vs 20 의 차이 중 얼마가 변경 효과인지 말할 수 없다.`
    : '\n→ 게이트는 결정적이다. 뒤집힌 문항은 청크 구성 변화로 설명해야 한다.',
)
