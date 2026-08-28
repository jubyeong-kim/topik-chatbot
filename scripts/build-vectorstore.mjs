// 청크를 임베딩해 public/vectorstore.json 을 만든다. (PRD v2 §8)
// 런타임에는 질문 임베딩만 계산하고, 청크 임베딩은 이 스크립트가 미리 만든다.
//
//   npm run build:vectorstore
//
// ⚠️ e5 계열은 접두어가 필수다. 청크는 "passage: ", 질문은 "query: " 를 붙인다.
//    붙이지 않으면 검색 품질이 눈에 띄게 떨어진다.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from '@huggingface/transformers'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const IN = resolve(ROOT, 'src/data/chunks.json')
const OUT = resolve(ROOT, 'public/vectorstore.json')

const MODEL = 'Xenova/multilingual-e5-small'
const DTYPE = 'q8'

const data = JSON.parse(readFileSync(IN, 'utf8'))
const { sources, chunks } = data

console.log(`청크 ${chunks.length}개 / 모델 ${MODEL} (${DTYPE})`)
console.log('모델을 처음 받을 때는 몇 분 걸립니다...')

const embed = await pipeline('feature-extraction', MODEL, { dtype: DTYPE })

const records = []
for (const [i, c] of chunks.entries()) {
  const prefix = c.id.split('-')[0]
  const src = sources[prefix]
  if (!src) throw new Error(`${c.id}: sources 에 "${prefix}" 정의가 없습니다`)

  // 긴 청크는 검색용 짧은 표현(index_text)으로 임베딩하고, 표시는 원문(text)으로 한다.
  // 442자짜리 표를 통째로 임베딩하면 mean pooling 에서 주제가 희석되기 때문이다.
  const out = await embed(`passage: ${c.index_text ?? c.text}`, { pooling: 'mean', normalize: true })

  records.push({
    id: c.id,
    text: c.text,
    category: c.category,
    source_title: src.title,
    source_url: src.url,
    official: src.official,
    exam_mode: src.exam_mode,
    // 소수 6자리면 코사인 결과가 바뀌지 않으면서 파일이 작아진다
    vector: Array.from(out.data, (v) => Math.round(v * 1e6) / 1e6),
  })

  if ((i + 1) % 10 === 0) console.log(`  ${i + 1}/${chunks.length}`)
}

mkdirSync(dirname(OUT), { recursive: true })
writeFileSync(
  OUT,
  JSON.stringify(
    { built_at: new Date().toISOString().slice(0, 10), model: MODEL, dtype: DTYPE, dim: records[0].vector.length, chunks: records },
    null,
    0,
  ),
)

const kb = Math.round(readFileSync(OUT).length / 1024)
console.log(`\n완료: ${OUT}`)
console.log(`${records.length}개 · ${records[0].vector.length}차원 · ${kb}KB`)
