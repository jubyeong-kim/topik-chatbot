// 브라우저에서 질문 하나를 임베딩한다. (PRD v3 §8)
//
// 청크 임베딩은 scripts/build-vectorstore.mjs 가 미리 만들어 두고,
// 런타임은 질문만 계산한다.
//
// ⚠️ 문서와 질문이 같은 공간에 있어야 거리가 뜻을 갖는다.
//    모델 ID·dtype·차원이 벡터스토어와 하나라도 다르면 숫자는 나와도 의미가 없다.
//    그래서 로드 전에 벡터스토어에 기록된 값을 그대로 쓰고, 계산 후 차원을 대조한다.

import { pipeline, env } from '@huggingface/transformers'
import type { Chunk } from './search'

env.allowLocalModels = false

export type Store = {
  built_at: string
  model: string
  dtype: string
  dim: number
  chunks: Chunk[]
}

export type LoadProgress =
  | { phase: 'store'; message: string }
  | { phase: 'model'; message: string; percent: number; cached: boolean }
  | { phase: 'ready'; message: string; cached: boolean; elapsedMs: number }

type Embedder = (text: string) => Promise<number[]>

let cache: { store: Store; embed: Embedder } | null = null

export async function load(
  baseUrl: string,
  onProgress: (p: LoadProgress) => void,
): Promise<{ store: Store; embed: Embedder }> {
  if (cache) return cache

  const t0 = performance.now()

  onProgress({ phase: 'store', message: '자료를 불러오는 중' })
  const res = await fetch(`${baseUrl}vectorstore.json`)
  if (!res.ok) throw new Error(`벡터스토어를 못 읽었습니다 (${res.status})`)
  const store: Store = await res.json()

  // 캐시에서 꺼냈는지 새로 받았는지는 **Cache Storage 를 직접 보고** 판단한다.
  //
  // ⚠️ progress_callback 으로는 판단할 수 없다. transformers.js 는 캐시에서 읽을 때도
  //    progress 이벤트를 내보내므로, 그것만 보면 캐시 적중을 다운로드로 오판한다.
  //    (실제로 그렇게 잘못 표시한 적이 있다.)
  //    캐시 여부는 재방문 경험의 계약이므로 틀리게 말하면 안 된다.
  const cachedBefore = await isModelCached()

  let percent = 0
  const extractor = await pipeline('feature-extraction', store.model, {
    dtype: store.dtype as 'q8',
    progress_callback: (p: { status: string; progress?: number; file?: string }) => {
      if (typeof p.progress === 'number') percent = Math.round(p.progress)
      onProgress({
        phase: 'model',
        message: cachedBefore ? '모델을 여는 중' : '임베딩 모델을 받는 중',
        percent,
        cached: cachedBefore,
      })
    },
  })

  const embed: Embedder = async (text: string) => {
    // e5 계열은 접두어가 필수다. 질문은 "query: ", 청크는 "passage: ".
    const out = await extractor(`query: ${text}`, { pooling: 'mean', normalize: true })
    const v = Array.from(out.data as Float32Array)

    // 공간 일치 검사 — 차원이 다르면 코사인이 무의미하다
    if (v.length !== store.dim) {
      throw new Error(
        `질문 벡터 ${v.length}차원 ≠ 벡터스토어 ${store.dim}차원. ` +
          `같은 모델(${store.model})로 만든 것이 맞는지 확인하세요.`,
      )
    }
    return v
  }

  // 워밍업 겸 위 검사를 즉시 한 번 태운다
  await embed('확인')

  onProgress({
    phase: 'ready',
    message: cachedBefore ? '준비 완료 (캐시 사용)' : '준비 완료',
    cached: cachedBefore,
    elapsedMs: Math.round(performance.now() - t0),
  })

  cache = { store, embed }
  return cache
}

/**
 * transformers.js 가 쓰는 Cache Storage 에 모델 파일이 이미 있는지 본다.
 * Cache Storage 는 시크릿 창 등에서 실패할 수 있으므로, 실패하면 "모른다"가 아니라
 * "캐시 없음"으로 처리한다 — 없다고 말하고 받는 편이, 있다고 말하고 기다리게 하는 것보다 낫다.
 */
async function isModelCached(): Promise<boolean> {
  try {
    if (!('caches' in globalThis)) return false
    const c = await caches.open('transformers-cache')
    const keys = await c.keys()
    return keys.some((r) => r.url.endsWith('.onnx'))
  } catch {
    return false
  }
}
