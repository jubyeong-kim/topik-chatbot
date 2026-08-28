import { readFileSync } from 'node:fs'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
const store = JSON.parse(readFileSync(new URL('../public/vectorstore.json', import.meta.url),'utf8'))
const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)
// 범위 안처럼 보이지만 자료에 답이 없는 질문들 (봉인 세트 아님 — 진단용)
const probes = [
  '54번에서 맞춤법을 하나 틀리면 몇 점이 깎이나요?',
  '53번 채점자는 몇 명인가요?',
  '쓰기 답안을 연필로 쓰면 감점인가요?',
  '54번에서 사자성어를 쓰면 가산점이 있나요?',
]
for (const q of probes) {
  const out = await embed(`query: ${q}`, { pooling:'mean', normalize:true })
  const r = search(store.chunks, index, q, Array.from(out.data))
  console.log(`[${r.verdict}] ${q}`)
  console.log(`   → ${r.hits.slice(0,2).map(h=>h.id).join(' ')} (cos ${r.topCosine.toFixed(3)})`)
}
