import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig, loadEnv } from 'vite'

// In dev the API is proxied, as nginx does in production, so the browser sees one origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxy = {
    '/tts-api': {
      target: env.TTS_API_URL || 'http://localhost:8888',
      changeOrigin: true,
      rewrite: (path: string) => path.replace(/^\/tts-api/, ''),
    },
  }
  return { plugins: [svelte()], server: { proxy }, preview: { proxy } }
})
