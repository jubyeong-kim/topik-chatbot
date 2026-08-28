// 선택 엔진(Gemini)의 속도를 로컬과 같은 조건에서 잰다. (PRD §9-B 비교표의 [미측정] 칸)
//
//   node scripts/bench-gemini.mjs
//
// 🔑 키는 이 파일에 적지 않는다. 아래 둘 중 하나로 준다.
//   1) 저장소 루트에 .env.local 파일을 만들고 한 줄:  GEMINI_API_KEY=...
//      (.gitignore 의 *.local 에 걸려 커밋되지 않는다)
//   2) 환경변수 GEMINI_API_KEY 로 준다
//
// 스크립트는 키를 화면에 찍지 않는다. 앞 4글자만 보여 어떤 키인지 확인만 한다.
//
// 부하는 bench-model.mjs 와 **같다** — 같은 질문 3개, 같은 GATE_K/ANSWER_K,
// 같은 프롬프트. 다르면 엔진 비교가 아니라 조건 비교가 된다.

import { readFileSync, existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'
import { GATE_K, ANSWER_K, GATE_SYSTEM, answerSystem } from '../src/lib/ollama.ts'
import { gate, answer, GEMINI_MODEL } from '../src/lib/gemini.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')

function readKey() {
  if (process.env.GEMINI_API_KEY) return process.env.GEMINI_API_KEY.trim()
  const f = resolve(ROOT, '.env.local')
  if (existsSync(f)) {
    const m = readFileSync(f, 'utf8').match(/^\s*GEMINI_API_KEY\s*=\s*(.+)\s*$/m)
    if (m) return m[1].trim().replace(/^["']|["']$/g, '')
  }
  return ''
}

const key = readKey()
if (!key) {
  console.error(`키를 찾지 못했습니다.

저장소 루트에 .env.local 파일을 만들고 한 줄 적으세요:
  GEMINI_API_KEY=여기에_본인_키

이 파일은 .gitignore 의 *.local 에 걸려 커밋되지 않습니다.`)
  process.exit(1)
}
console.log(`키 확인됨 (${key.slice(0, 4)}…, ${key.length}자) · 모델 ${GEMINI_MODEL}\n`)

const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const QUESTIONS = [
  '54번은 어떤 기준으로 채점되나요?',
  '53번에서 감점되는 경우는 뭔가요?',
  '3급과 5급의 평가 기준은 뭐가 다른가요?',
]

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

const rows = []
for (const q of QUESTIONS) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const hits = search(store.chunks, index, q, Array.from(out.data)).hits

  const g0 = Date.now()
  const g = await gate(q, hits.slice(0, GATE_K), GATE_SYSTEM, { key })
  const gateMs = Date.now() - g0

  const t0 = Date.now()
  let firstMs = null, text = ''
  for await (const piece of answer(q, hits.slice(0, ANSWER_K), answerSystem(false), { key })) {
    if (firstMs === null) firstMs = Date.now() - t0
    text += piece
  }
  const totalMs = Date.now() - t0
  rows.push({ gateMs, firstMs, totalMs, len: text.length })
  console.log(
    `✓ 게이트 ${(gateMs / 1000).toFixed(1)}초 (${g.raw.trim()}) · 첫 토큰 ${(firstMs / 1000).toFixed(1)}초 · ` +
      `전체 ${(totalMs / 1000).toFixed(1)}초 · 답변 ${text.length}자`,
  )
}

const avg = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length
console.log(
  `\n평균 게이트 ${(avg((r) => r.gateMs) / 1000).toFixed(1)}초 · 첫 토큰 ${(avg((r) => r.firstMs) / 1000).toFixed(1)}초` +
    `\n질문 하나당 게이트+생성 ${((avg((r) => r.gateMs) + avg((r) => r.totalMs)) / 1000).toFixed(1)}초` +
    `\n\n비교 — 로컬 qwen2.5:1.5b 는 질문당 48.8초, 첫 토큰 11.6초였다 (scripts/bench-model.mjs)`,
)
