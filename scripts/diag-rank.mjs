// 스팟체크: "정답 청크가 상위에 오는가". 거절/답변 여부와 별개의 지표다.
// 기대 청크는 코퍼스 작성 후에 붙인 주석이며, 질문 자체는 봉인본 그대로다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search, cosine } from '../src/lib/search.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))

// 질문 → 이 청크가 상위에 와야 한다 (하나라도 오면 성공)
const EXPECT = {
  '54번은 어떤 기준으로 채점되나요?': ['TPK-024', 'TPK-025'],
  '53번과 54번의 배점 차이는 얼마인가요?': ['NUM-005', 'TPK-012', 'TPK-022'],
  '쓰기에서 감점되는 경우는 어떤 것들이 있나요?': ['TPK-005', 'TPK-006', 'TPK-007', 'TPK-020', 'TPK-029'],
  '글의 제목을 쓰면 감점되나요?': ['TPK-012'],
  "54번에서 '-습니다'체를 쓰면 감점인가요?": ['TPK-029', 'EXP-007'],
  '3급과 5급의 평가 기준은 뭐가 다른가요?': ['LVL-003', 'LVL-005'],
  '조사를 빠뜨리면 채점에 어떻게 반영되나요?': ['GRM-001', 'GRM-002', 'EXP-001'],
  '글자 수를 못 채우면 어떻게 되나요?': ['TPK-012', 'TPK-017', 'TPK-022'],
  "'내용 및 과제 수행'은 몇 점인가요?": ['TPK-014', 'TPK-024'],
  "'언어 사용' 범주에서는 무엇을 보나요?": ['TPK-014', 'TPK-024', 'EXP-002'],
  'IBT와 PBT의 쓰기는 무엇이 다른가요?': ['EXM-005', 'EXM-001'],
  '쓰기 시험 시간은 얼마인가요?': ['EXM-001', 'EXM-003', 'NUM-003'],
  '51번과 52번은 어떻게 다른가요?': ['TPK-002', 'TPK-010'],
  '띄어쓰기 실수도 채점에 반영되나요?': ['GRM-004', 'EXP-001'],
  '6급을 받으려면 몇 점이 필요한가요?': ['NUM-001', 'EXM-004'],
}

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

// 합집합 검색에서 중요한 것은 "정답 청크가 최종 후보에 들어왔는가"(recall)다.
// 그 다음이 코사인 상위권 순위다.
let recall = 0, vTop1 = 0, vTop3 = 0, onlyBm25 = 0
console.log('질문                                            후보  코사인순위  들어온경로   실제 top1')
console.log('-'.repeat(96))

for (const [q, want] of Object.entries(EXPECT)) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const v = Array.from(out.data)

  const r = search(store.chunks, index, q, v, { bm25B: Number(process.argv[2] ?? 0.75) })
  const found = want.map((id) => r.hits.find((h) => h.id === id)).filter(Boolean)

  const vecHits = r.hits.filter((h) => h.method === 'vector')
  const ranks = want.map((id) => vecHits.findIndex((h) => h.id === id) + 1).filter((n) => n > 0)
  const best = ranks.length ? Math.min(...ranks) : Infinity

  if (found.length) recall++
  if (best === 1) vTop1++
  if (best <= 3) vTop3++
  if (found.length && best === Infinity) onlyBm25++

  const via = !found.length ? '없음' : best !== Infinity ? 'vector' : 'bm25만'
  console.log(
    `${q.slice(0, 44).padEnd(46)}${(found.length ? '있음' : '없음').padEnd(6)}` +
      `${String(best === Infinity ? '-' : best).padStart(6)}      ${via.padEnd(10)} ${r.hits[0].id}`,
  )
}

const n = Object.keys(EXPECT).length
console.log(`\n후보 포함(recall) ${recall}/${n} · 코사인 top-1 ${vTop1}/${n} · top-3 ${vTop3}/${n} · BM25 가 살린 것 ${onlyBm25}/${n}`)

// 순위를 코사인만으로 매기면 어떻게 되는지 비교 (BM25 가중치가 도움이 되는지 확인)
console.log('\n[비교] 코사인 단독 순위')
let c1 = 0, c3 = 0
for (const [q, want] of Object.entries(EXPECT)) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const v = Array.from(out.data)
  const ranked = store.chunks
    .map((c) => ({ id: c.id, s: cosine(c.vector, v) }))
    .sort((a, b) => b.s - a.s)
  const ranks = want.map((id) => ranked.findIndex((h) => h.id === id) + 1).filter((x) => x > 0)
  const best = Math.min(...ranks)
  if (best === 1) c1++
  if (best <= 3) c3++
}
console.log(`top-1 적중 ${c1}/${n} · top-3 ${c3}/${n}`)
