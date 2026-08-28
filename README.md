# TOPIK 채점 안내봇

TOPIK 쓰기가 **어떻게 채점되는지**를 공개 자료에 근거해 답하고, **근거를 함께 보여주는** 브라우저 RAG 챗봇.

👉 **[jubyeong-kim.github.io/topik-chatbot](https://jubyeong-kim.github.io/topik-chatbot/)**

---

## ⚠️ 배포 주소를 열어도 답변이 안 나오는 것이 정상입니다

정적 페이지를 올린 것은 **모델을 인터넷 서버에 올린 것이 아닙니다.**
배포된 앱은 방문자의 브라우저에서 실행되고, **방문자 자신의 컴퓨터**에 있는 `localhost:11434` 를 부릅니다.

| | Ollama 없이 | Ollama 설치 후 |
|---|---|---|
| 검색 · 근거 표시 · 거절 | ✅ 됩니다 | ✅ |
| 답변 생성 | ❌ 안 됩니다 | ✅ |

**검색과 거절은 누구나 볼 수 있습니다.** 임베딩이 브라우저 안에서 돌기 때문입니다.
답변 생성만 각자의 로컬 모델이 필요합니다.

> 답변까지 보는 길은 두 가지입니다 — **Ollama 설치**(아무것도 밖으로 안 나감)
> 또는 **본인 Gemini 키**(질문과 근거가 Google 로 나감). 아래에 둘 다 적었습니다.

### 답변까지 보려면

1. [ollama.com/download](https://ollama.com/download) 에서 설치
2. 모델 받기

```bash
ollama pull qwen2.5:1.5b-instruct-q4_K_M
```

3. **배포 주소에서 쓰려면** 출처를 허용해야 합니다. Ollama 는 기본적으로 localhost 에서
   서빙된 페이지만 허용하므로, `https://...github.io` 출처는 거부됩니다.

```bash
setx OLLAMA_ORIGINS "https://jubyeong-kim.github.io"
```

설정 후 **Ollama 를 재시작**하세요. 로컬(`npm run dev`)에서는 이 설정이 필요 없습니다.

- **Chrome · Edge 권장.** Safari 는 https 페이지에서 `http://localhost` 를 mixed content 로 차단합니다

### 또는 — Gemini API 를 켜면 설치 없이도 답변이 나옵니다

화면의 **답변 엔진**에서 `Gemini API` 를 고르고 본인 키를 넣으면 Ollama 없이도 답변이 생성됩니다.

⚠️ **켜면 질문과 근거 청크가 Google 서버로 나갑니다.** 로컬 Ollama 를 쓸 때는 아무것도 나가지 않습니다.
어느 쪽인지 화면에 계속 표시됩니다.

🔑 **키는 이 저장소에도 배포 파일에도 들어 있지 않습니다.** 사용자가 직접 넣고 그 브라우저에만 남습니다.
정적 사이트는 서버가 없어 키를 숨길 곳이 없으므로, 번들에 키를 넣는 방식은 쓰지 않았습니다.
키 발급: [Google AI Studio](https://aistudio.google.com/apikey)

두 엔진은 **같은 프롬프트와 같은 근거 개수**를 씁니다. 다르면 엔진 비교가 아니라 프롬프트 비교가 됩니다.
답변 아래에 `엔진명 · 게이트+생성 N초` 가 표시되어 응답 리듬을 비교할 수 있습니다.

---

## 무엇을 보여주려고 만들었나

답을 잘 내는 것보다 **답이 어디서 왔는지 보이게** 하는 데 무게를 뒀습니다.

- 답변 아래에 **근거로 쓴 자료**가 항상 붙습니다
- **파란 칩**은 공식 자료, **회색 칩 + "비공식"** 은 자체 해설입니다
- 각 근거에 **어느 검색기로 들어왔는지**(`vector` / `bm25`)와 점수가 붙습니다
- 자료에 답이 없으면 **답하지 않고 거절합니다.** 왜 거절했는지도 밝힙니다

## 거절은 3층입니다

| 층 | 무엇 | 성격 |
|---|---|---|
| 1층 범위 규칙 | 쓰기 외 영역 · 시험 운영 · 진로 · 교수법 · 첨삭 요청 | 결정적 |
| 2층 LLM 게이트 | *"이 자료로 답할 수 있는가"* 를 모델에 묻는다 | 확률적 |
| 3층 약한 근거 | 어휘 일치가 없으면 배지. **거절이 아니라 표시** | 보조 |

**코사인 임계값으로 거절하는 방식은 폐기했습니다.** 자료 안 질문의 top1 코사인이 0.838~0.900,
자료 밖이 0.850~0.875 로 **완전히 겹쳤기 때문입니다.** 자료 밖 질문도 전부 TOPIK 이야기라
임베딩이 재는 "주제 근접성"으로는 *"이 청크가 이 질문의 답이 되는가"* 를 가릴 수 없습니다.

## 실측

| 항목 | 값 |
|---|---|
| 거절 정확도 (25문항) | **21/25** |
| **과소거절** (자료 밖인데 답함) | **0** |
| 검색 recall | 14/15 |
| 질문 하나당 | 콜드 약 50초 · 웜 약 35초 |
| 첫 방문 모델 준비 | 39초 (약 120MB 다운로드, 이후 캐시) |

느립니다. 8GB RAM 에서 CPU 로 1.5B 를 돌리는 대가입니다.
`qwen2.5:3b` 도 재봤지만 질문당 165초 + 스왑 2.6GB 로 기각했습니다.

## 구조

```
src/data/chunks.json          청크 65개 (공식 46 / 비공식 19)
src/lib/search.ts             코사인 10개 + BM25 5개 합집합, 범위 규칙
src/lib/embed.ts              브라우저 질문 임베딩, 공간 일치 검사
src/lib/ollama.ts             게이트 · 답변 스트리밍
src/lib/judge.ts              판정 (셀 수 있는 것은 코드로 계산)
scripts/build-vectorstore.mjs 청크 → public/vectorstore.json
eval/questions.json           봉인 질문 20개
```

### 개발

```bash
npm install
npm run build:vectorstore   # 청크가 바뀌었을 때만
npm run dev
```

### 측정

```bash
node scripts/eval-gate.mjs       # 거절 정확도 25문항
node scripts/diag-rank.mjs       # 검색 recall · 순위
node scripts/bench-model.mjs qwen2.5:1.5b-instruct-q4_K_M
```

측정 스크립트는 앱과 **같은 상수**(`GATE_K` · `ANSWER_K`)를 씁니다.
한때 앱은 8개, 스크립트는 5개를 넣어 측정과 동작이 달랐습니다.

## 자료 출처

| 접두어 | 자료 |
|---|---|
| `TPK-` | [TOPIK Ⅱ 쓰기 답안 작성 방법](https://exam.topik.go.kr/) — 국립국제교육원 |
| `EXM-` | [국립국제교육원 TOPIK IBT 응시자 안내서](https://www.niied.go.kr/web/niied/contents/niied_topik) |
| `LVL-` | [TOPIK 등급별 평가 기준](https://www.topik.go.kr/) |
| `GRM-` `NUM-` `EXP-` | 자체 작성 · 공개 정보 정리 · 자체 해설 (**비공식**) |

> TOPIK 은 **국립국제교육원(NIIED)** 이 시행합니다.
> 이 앱은 공식 서비스가 아니며, 학습 목적의 실습 결과물입니다.
> 정확한 내용은 반드시 [topik.go.kr](https://www.topik.go.kr/) 에서 확인하세요.
