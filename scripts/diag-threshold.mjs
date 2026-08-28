// 왜 거절이 안 되는지 진단한다. 질문별 코사인 '분포'를 본다.
// 가설: e5 는 절대값 범위가 좁아 임계값으로 자를 수 없고,
//       자료 안 질문은 특정 청크가 튀어나오는 반면 자료 밖 질문은 분포가 평평하다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { cosine } from '../src/lib/search.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const evalSet = JSON.parse(readFileSync(resolve(ROOT, 'eval/questions.json'), 'utf8'))

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })

console.log('id   expect      top1   mean   std    top1-mean  z(top1)  top1-top5')
console.log('-'.repeat(72))

const rows = []
for (const q of evalSet.questions) {
  const out = await embed(`query: ${q.q}`, { pooling: 'mean', normalize: true })
  const v = Array.from(out.data)
  const cs = store.chunks.map((c) => cosine(c.vector, v)).sort((a, b) => b - a)

  const mean = cs.reduce((a, b) => a + b, 0) / cs.length
  const std = Math.sqrt(cs.reduce((a, b) => a + (b - mean) ** 2, 0) / cs.length)
  const row = {
    id: q.id,
    expect: q.expect,
    top1: cs[0],
    mean,
    std,
    gap: cs[0] - mean,
    z: (cs[0] - mean) / std,
    spread: cs[0] - cs[4],
  }
  rows.push(row)
  console.log(
    `${row.id}  ${row.expect.padEnd(10)} ${row.top1.toFixed(3)}  ${mean.toFixed(3)}  ${std.toFixed(3)}  ` +
      `${row.gap.toFixed(3)}      ${row.z.toFixed(2)}     ${row.spread.toFixed(3)}`,
  )
}

const inC = rows.filter((r) => r.expect === 'in_corpus')
const outC = rows.filter((r) => r.expect === 'refuse')
const rng = (a, f) => `${Math.min(...a.map(f)).toFixed(3)} ~ ${Math.max(...a.map(f)).toFixed(3)}`

console.log('\n지표별 분리 가능성 (겹치면 그 지표로는 못 자른다)')
for (const [name, f] of [['top1', (r) => r.top1], ['gap', (r) => r.gap], ['z', (r) => r.z], ['spread', (r) => r.spread]]) {
  const lo = Math.min(...inC.map(f))
  const hi = Math.max(...outC.map(f))
  console.log(`  ${name.padEnd(7)} 자료안 ${rng(inC, f)} | 자료밖 ${rng(outC, f)}  →  ${lo > hi ? `분리 가능 (경계 ${((lo + hi) / 2).toFixed(3)})` : '겹침'}`)
}
