// 봉인 질문 20개를 검색에만 통과시켜 거절 판정을 측정한다. LLM 없이 돌아간다.
//
//   node scripts/eval-retrieval.mjs [코사인임계값]
//
// 판정: expect=in_corpus 인데 refuse 면 과잉거절, expect=refuse 인데 answer/weak 면 과소거절.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search, DEFAULTS } from '../src/lib/search.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const evalSet = JSON.parse(readFileSync(resolve(ROOT, 'eval/questions.json'), 'utf8'))

const threshold = Number(process.argv[2] ?? DEFAULTS.cosineThreshold)

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

console.log(`청크 ${store.chunks.length}개 · 코사인 임계값 ${threshold}\n`)

let ok = 0
const wrong = []

for (const q of evalSet.questions) {
  const out = await embed(`query: ${q.q}`, { pooling: 'mean', normalize: true })
  const r = search(store.chunks, index, q.q, Array.from(out.data), { cosineThreshold: threshold })

  const answered = r.verdict !== 'refuse'
  const shouldAnswer = q.expect === 'in_corpus'
  const pass = answered === shouldAnswer
  if (pass) ok++
  else wrong.push({ ...q, verdict: r.verdict, topCosine: r.topCosine })

  const mark = pass ? '✓' : '✗'
  const ids = r.hits.slice(0, 3).map((h) => h.id).join(' ')
  console.log(
    `${mark} ${q.id} [${r.verdict.padEnd(6)}] cos=${r.topCosine.toFixed(3)} ${ids}`,
  )
  console.log(`   ${q.q}`)
}

console.log(`\n정확: ${ok}/${evalSet.questions.length}`)
if (wrong.length) {
  console.log('\n어긋난 것:')
  for (const w of wrong) {
    const kind = w.expect === 'in_corpus' ? '과잉거절' : '과소거절'
    console.log(`  ${w.id} ${kind} (expect=${w.expect}, got=${w.verdict}, cos=${w.topCosine.toFixed(3)})  ${w.q}`)
  }
}
