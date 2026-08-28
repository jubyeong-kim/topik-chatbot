// 게이트 지연을 다시 잰다. 앞 측정은 두 가지가 틀렸다.
//   1) 3문항 평균을 내서 개별 실행의 편차가 가려졌다
//   2) 같은 프롬프트를 반복해 Ollama 의 KV 캐시가 섞였다
//
// 이번에는 설정마다 **서로 다른 질문**을 쓰고, **개별 실행을 그대로** 찍는다.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'
import { buildBm25Index, search } from '../src/lib/search.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const store = JSON.parse(readFileSync(resolve(ROOT, 'public/vectorstore.json'), 'utf8'))
const evalSet = JSON.parse(readFileSync(resolve(ROOT, 'eval/questions.json'), 'utf8'))
const OLLAMA = 'http://localhost:11434'
const MODEL = 'qwen2.5:1.5b-instruct-q4_K_M'

// 자료 안 질문 12개를 4개씩 세 묶음으로 나눠, 설정마다 다른 질문을 쓴다
const pool = evalSet.questions.filter((x) => x.expect === 'in_corpus').map((x) => x.q)

const GATE_SYSTEM = `당신은 자료 검토자입니다. 답을 만들지 마십시오.
주어진 자료만 보고, 질문에 답할 근거가 자료 안에 있는지만 판단합니다.

YES: 자료에 적힌 내용만으로 질문에 답할 수 있다.
NO: 자료가 주제만 비슷할 뿐, 질문이 묻는 사실이 자료에 없다.

당신의 배경지식은 쓰지 마십시오. 자료에 없으면 NO 입니다.
YES 또는 NO 한 단어만 출력하십시오.`

const embed = await pipeline('feature-extraction', store.model, { dtype: store.dtype })
const index = buildBm25Index(store.chunks)

async function gateOnce(q, chunkCount, numPredict) {
  const out = await embed(`query: ${q}`, { pooling: 'mean', normalize: true })
  const hits = search(store.chunks, index, q, Array.from(out.data)).hits.slice(0, chunkCount)
  const evidence = hits.map((h, i) => `[${i + 1}] (${h.id}) ${h.text}`).join('\n')
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: false, options: { temperature: 0, num_predict: numPredict },
      messages: [
        { role: 'system', content: GATE_SYSTEM },
        { role: 'user', content: `자료:\n${evidence}\n\n질문: ${q}\n\nYES 또는 NO:` },
      ],
    }),
  })
  const j = await res.json()
  return {
    total: j.total_duration / 1e9,
    pTok: j.prompt_eval_count ?? 0,
    pSec: (j.prompt_eval_duration ?? 0) / 1e9,
    eTok: j.eval_count ?? 0,
    eSec: (j.eval_duration ?? 0) / 1e9,
    text: (j.message?.content ?? '').replace(/\s+/g, ''),
  }
}

async function run(label, qs, chunkCount, numPredict) {
  console.log(`\n${label}`)
  const totals = []
  for (const q of qs) {
    const r = await gateOnce(q, chunkCount, numPredict)
    totals.push(r.total)
    console.log(
      `  ${r.total.toFixed(1)}초  프롬프트 ${r.pTok}토큰/${r.pSec.toFixed(1)}초 ` +
        `(${(r.pTok / Math.max(r.pSec, 0.001)).toFixed(0)} tok/s) · 생성 ${r.eTok}토큰/${r.eSec.toFixed(1)}초 · ${r.text}  ${q.slice(0, 22)}`,
    )
  }
  const min = Math.min(...totals), max = Math.max(...totals)
  console.log(`  → ${min.toFixed(1)}~${max.toFixed(1)}초 (중앙 ${totals.sort((a, b) => a - b)[Math.floor(totals.length / 2)].toFixed(1)})`)
  return totals
}

await gateOnce(pool[0], 8, 1) // 워밍업 (모델 적재)

await run('[A] 청크 8개 · num_predict 5', pool.slice(1, 5), 8, 5)
await run('[B] 청크 8개 · num_predict 1', pool.slice(5, 9), 8, 1)
await run('[C] 청크 4개 · num_predict 5', pool.slice(9, 13), 4, 5)

console.log(`
읽는 법
  프롬프트 tok/s 가 설정마다 비슷하면 → 지연은 '읽을 토큰 수'에 비례한다 → 청크 수가 레버다
  생성 토큰이 1~2개뿐이면 → num_predict 를 줄여도 아낄 시간이 거의 없다`)
