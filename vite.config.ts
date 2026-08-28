import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages 는 데모용이라 base 가 저장소 이름이어야 한다 (PRD v2 §4).
// 로컬 개발에서는 '/' 로 두어야 dev 서버 경로가 꼬이지 않는다.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/topik-chatbot/' : '/',
}))
