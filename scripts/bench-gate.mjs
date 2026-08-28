// 게이트가 왜 느린지 분해한다. 한 번에 하나씩만 바꾼다.
//
// Ollama 응답에는 시간이 나뉘어 들어온다.
//   prompt_eval_duration : 프롬프트(청크 8개)를 읽는 시간
//   eval_duration        : 답을 만드는 시간
// 어느 쪽이 지배적인지 알면 무엇을 줄여야 하는지가 정해진다.
//
//   node scripts/bench-gate.mjs

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const OLLAMA = 'http://localhost:11434'
const MODEL = 'qwen2.5:1.5b-instruct-q4_K_M'

const QUESTIONS = [
  '54번은 어떤 기준으로 채점되나요?',
  '53번에서 감점되는 경우는 뭔가요?',
  '3급과 5급의 평가 기준은 뭐가 다른가요?',
]

const GATE_SYSTEM = `당신은 자료 검토자입니다. 답을 만들지 마십시오.
주어진 자료만 보고, 질문에 답할 근거가 자료 안에 있는지만 판단합니다.

YES: 자료에 적힌 내용만으로 질문에 답할 수 있다.
NO: 자료가 주제만 비슷할 뿐, 질문이 묻는 사실이 자료에 없다.

당신의 배경지식은 쓰지 마십시오. 자료에 없으면 NO 입니다.
YES 또는 NO 한 단어만 출력하십시오.`

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

const prepared = []
for (const q of QUESTIONS) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  prepared.push({ q, hits: search(store.chunks, index, q, Array.from(out.data)).hits })
}

async function gateOnce(q, hits, numPredict) {
  const evidence = hits.map((h, i) => `[${i + 1}] (${h.id}) ${h.text}`).join('\n')
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: false,
      options: { temperature: 0, num_predict: numPredict },
      messages: [
        { role: 'system', content: GATE_SYSTEM },
        { role: 'user', content: `자료:\n${evidence}\n\n질문: ${q}\n\nYES 또는 NO:` },
      ],
    }),
  })
  const j = await res.json()
  return {
    total: j.total_duration / 1e9,
    load: (j.load_duration ?? 0) / 1e9,
    promptTokens: j.prompt_eval_count ?? 0,
    promptSec: (j.prompt_eval_duration ?? 0) / 1e9,
    evalTokens: j.eval_count ?? 0,
    evalSec: (j.eval_duration ?? 0) / 1e9,
    text: (j.message?.content ?? '').trim(),
  }
}

async function run(label, numPredict, chunkCount) {
  const rows = []
  for (const { q, hits } of prepared) rows.push(await gateOnce(q, hits.slice(0, chunkCount), numPredict))
  const avg = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length
  console.log(
    `${label.padEnd(28)} 총 ${avg((r) => r.total).toFixed(1)}초  ` +
      `= 로드 ${avg((r) => r.load).toFixed(1)} + 프롬프트 ${avg((r) => r.promptSec).toFixed(1)}(${Math.round(avg((r) => r.promptTokens))}토큰) ` +
      `+ 생성 ${avg((r) => r.evalSec).toFixed(1)}(${Math.round(avg((r) => r.evalTokens))}토큰)  ` +
      `판정 ${rows.map((r) => r.text.replace(/\s+/g, '')).join('/')}`,
  )
  return avg((r) => r.total)
}

console.log('워밍업(모델 적재)…')
await gateOnce(prepared[0].q, prepared[0].hits.slice(0, 8), 1)

console.log('\n[기준] 청크 8개 · num_predict 5')
const base = await run('청크8 · predict5', 5, 8)

console.log('\n[변경 1] num_predict 만 1 로')
const p1 = await run('청크8 · predict1', 1, 8)

console.log('\n[변경 2] 청크 수만 5 로 (predict 는 5 유지)')
const c5 = await run('청크5 · predict5', 5, 5)

console.log('\n[변경 3] 청크 3개 (predict 5 유지)')
const c3 = await run('청크3 · predict5', 5, 3)

console.log(
  `\n기준 대비` +
    `\n  num_predict 1  : ${(((base - p1) / base) * 100).toFixed(0)}% 단축` +
    `\n  청크 5개       : ${(((base - c5) / base) * 100).toFixed(0)}% 단축` +
    `\n  청크 3개       : ${(((base - c3) / base) * 100).toFixed(0)}% 단축`,
)

// ── 대조 실험 ────────────────────────────────────────────────
// 위 비교는 순서 효과에 오염됐을 수 있다. Ollama 는 같은 접두사의 KV 를 캐시하므로
// 먼저 돈 설정만 값을 치르고 뒤는 공짜로 보인다.
// 기준 설정을 **맨 뒤에 한 번 더** 돌려 본다. 여기서 빨라지면 캐시가 원인이다.
console.log('\n[대조] 기준 설정(청크8·predict5)을 맨 뒤에 다시')
const again = await run('청크8 · predict5 (재실행)', 5, 8)
console.log(
  `\n첫 실행 ${base.toFixed(1)}초 → 재실행 ${again.toFixed(1)}초` +
    (again < base * 0.5
      ? '\n→ 같은 설정인데 빨라졌다. 위 "단축률" 은 변수 효과가 아니라 프롬프트 캐시다. 비교 무효.'
      : '\n→ 재실행도 느리다. 위 비교는 캐시 때문이 아니다.'),
)
