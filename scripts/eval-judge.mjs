// 판정기를 측정한다. 질문 몇 개를 끝까지(검색→게이트→답변→판정) 돌린다.
//
// 특히 볼 것: JSON 강제가 1.5B 의 판정을 망가뜨리는가.
// 게이트에서는 스키마 강제가 10/15 → 5/15 로 정확도를 떨어뜨렸다.
//
// cited 는 정규식으로도 셀 수 있으므로, 모델 판정과 실제 계산을 나란히 놓고
// 판정기가 사실을 맞히는지 확인한다. 이게 판정기를 판정하는 유일한 객관적 축이다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { answer, gate, health } from '../src/lib/ollama.ts'
import { judge, citedIds } from '../src/lib/judge.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))

const h = await health()
if (!h.ok) { console.error(h.reason); process.exit(1) }

const QUESTIONS = [
  { q: '54번은 어떤 기준으로 채점되나요?', kind: '정상' },
  { q: '3급과 5급의 평가 기준은 뭐가 다른가요?', kind: '정상' },
  { q: '쓰기 답안을 연필로 쓰면 감점인가요?', kind: '경계(자료 없음)' },
  { q: 'TOPIK 응시료는 얼마인가요?', kind: '무근거' },
]

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

for (const { q, kind } of QUESTIONS) {
  console.log('\n' + '='.repeat(78))
  console.log(`[${kind}] ${q}`)

  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const r = search(store.chunks, index, q, Array.from(out.data))

  if (r.verdict === 'refuse') { console.log(`→ 1층 거절: ${r.reason} (판정 대상 아님)`); continue }

  const top = r.hits.slice(0, 8)
  const g = await gate(q, top)
  if (!g.answerable) { console.log('→ 2층 거절: 검색된 자료에 답 없음 (판정 대상 아님)'); continue }

  let reply = ''
  for await (const piece of answer(q, top)) reply += piece
  console.log(`\n답변(${reply.length}자):\n${reply.slice(0, 260)}${reply.length > 260 ? '…' : ''}`)

  const real = citedIds(reply, top)
  const j = await judge(q, top, reply)

  if (!j.ok) { console.log(`\n판정 실패: ${j.error} / raw="${j.raw.slice(0, 120)}"`); continue }
  const v = j.verdict
  console.log(
    `\n판정  grounded=${v.grounded} noHalluc=${v.noHalluc} refusal=${v.refusal}  ← 모델` +
      `\n계산  cited=${v.cited} (${v.citedIds.join(', ') || '표기 없음'}) score=${v.score}  ← 코드` +
      `\n평어  ${v.comment}` +
      `\n대조  정규식으로 센 [ID]: ${real.join(', ') || '없음'}`,
  )
}
