// 모델 하나를 우리 실제 파이프라인 부하로 돌려 보고, 8GB 에서 쓸 만한지 판정한다.
//
//   node scripts/bench-model.mjs qwen2.5:3b-instruct-q4_K_M
//
// 재는 것
//   - 첫 토큰까지 걸린 시간 (사용자가 빈 화면을 보는 시간)
//   - 초당 토큰 수 (답변이 자라나는 속도)
//   - 게이트 왕복 시간 (질문마다 반드시 한 번 더 든다)
//   - 실패 여부 (메모리 부족이면 여기서 터진다)
//
// 부하는 앱과 같게 맞춘다 — 게이트 GATE_K 개, 생성 ANSWER_K 개.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { GATE_K, ANSWER_K } from '../src/lib/ollama.ts'

const MODEL = process.argv[2]
if (!MODEL) { console.error('모델 이름을 주세요'); process.exit(1) }

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const OLLAMA = 'http://localhost:11434'

const QUESTIONS = [
  '54번은 어떤 기준으로 채점되나요?',
  '53번에서 감점되는 경우는 뭔가요?',
  '3급과 5급의 평가 기준은 뭐가 다른가요?',
]

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

const evidence = (hits) => hits.map((h, i) => `[${i + 1}] (${h.id}) ${h.text}`).join('\n')

console.log(`모델: ${MODEL}\n게이트 ${GATE_K}개 · 생성 ${ANSWER_K}개 · ${QUESTIONS.length}문항.\n`)

const rows = []
for (const q of QUESTIONS) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const hits = search(store.chunks, index, q, Array.from(out.data)).hits
  const top = hits.slice(0, GATE_K)      // 게이트용
  const topAns = hits.slice(0, ANSWER_K) // 생성용

  // 게이트 왕복
  const g0 = Date.now()
  const gr = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: false, options: { temperature: 0, num_predict: 5 },
      messages: [
        { role: 'system', content: '자료로 답할 수 있으면 YES, 없으면 NO. 한 단어만.' },
        { role: 'user', content: `자료:\n${evidence(top)}\n\n질문: ${q}\n\nYES 또는 NO:` },
      ],
    }),
  })
  if (!gr.ok) { console.log(`✗ 게이트 실패 (${gr.status}) — ${q}`); rows.push(null); continue }
  await gr.json()
  const gateMs = Date.now() - g0

  // 답변 스트리밍
  const t0 = Date.now()
  let firstMs = null, tokens = 0, text = ''
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: true, options: { temperature: 0.2 },
      messages: [
        { role: 'system', content: '주어진 자료에 적힌 내용만으로 한국어로 답하십시오.' },
        // 생성은 ANSWER_K 개를 쓴다. 게이트용 top(GATE_K개)을 넣으면 앱과 부하가 달라진다.
        { role: 'user', content: `자료:\n${evidence(topAns)}\n\n질문: ${q}` },
      ],
    }),
  })
  if (!res.ok || !res.body) { console.log(`✗ 생성 실패 (${res.status}) — ${q}`); rows.push(null); continue }

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n'); buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      const j = JSON.parse(line)
      const p = j.message?.content
      if (p) { if (firstMs === null) firstMs = Date.now() - t0; tokens++; text += p }
    }
  }
  const totalMs = Date.now() - t0
  const tps = tokens / (totalMs / 1000)
  rows.push({ q, gateMs, firstMs, tokens, totalMs, tps })
  console.log(
    `✓ 게이트 ${(gateMs / 1000).toFixed(1)}초 · 첫 토큰 ${(firstMs / 1000).toFixed(1)}초 · ` +
      `${tokens}토큰 ${(totalMs / 1000).toFixed(1)}초 (${tps.toFixed(1)} tok/s) · 답변 ${text.length}자`,
  )
}

const ok = rows.filter(Boolean)
if (!ok.length) { console.log('\n전부 실패했습니다. 이 모델은 이 환경에서 쓸 수 없습니다.'); process.exit(1) }

const avg = (f) => ok.reduce((a, r) => a + f(r), 0) / ok.length
const 질문당 = (avg((r) => r.gateMs) + avg((r) => r.totalMs)) / 1000
console.log(
  `\n성공 ${ok.length}/${QUESTIONS.length}` +
    `\n평균 게이트 ${(avg((r) => r.gateMs) / 1000).toFixed(1)}초 · 첫 토큰 ${(avg((r) => r.firstMs) / 1000).toFixed(1)}초 · ${avg((r) => r.tps).toFixed(1)} tok/s` +
    `\n질문 하나당 게이트+생성 ${질문당.toFixed(1)}초 (판정은 별도)`,
)
