import { SseParser } from './sse'

const BASE = '/tts-api'

export interface ReferenceFile {
  filename: string
  source: string
  description: string | null
}

export interface Preset {
  name: string
  voice_description: string
  description?: string | null
}

export interface Voices {
  files: ReferenceFile[]
  presets: Preset[]
}

export interface Status {
  model_state: string
  model_id: string | null
  cuda?: { free_mb: number; total_mb: number }
}

export interface Info {
  supported_languages: string[]
}

export interface TtsRequest {
  text: string
  user: string
  language: string
  mode: 'clone' | 'design'
  model_size: string
  cpu_mode: boolean
  audio_format: 'wav'
  reference_audio?: string
  voice_preset?: string
  voice_description?: string
}

export interface Progress {
  percent: number
  chunk: number
  of: number
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const { detail } = await response.json()
    return typeof detail === 'string' ? detail : JSON.stringify(detail)
  } catch {
    return `The server answered ${response.status}`
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json()
}

const jsonBody = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

const stem = (filename: string) => filename.replace(/\.\w+$/, '')

// A voice's display name. Keeps the extension only when two files would otherwise look the same.
export function fileLabel(filename: string, files: ReferenceFile[]): string {
  const clash = files.some((file) => file.filename !== filename && stem(file.filename) === stem(filename))
  return clash ? filename : stem(filename)
}

const forUser = (user: string) => `?user=${encodeURIComponent(user)}`

export const getStatus = () => call<Status>('/status')
export const getInfo = () => call<Info>('/info')

export async function listVoices(user: string): Promise<Voices> {
  const [files, presets] = await Promise.all([
    call<{ files: ReferenceFile[] }>(`/reference-audio${forUser(user)}`),
    call<{ presets: Preset[] }>(`/voice-presets${forUser(user)}`),
  ])
  return {
    // The API keeps its preset store in the same folder as the audio.
    files: files.files.filter((file) => !file.filename.endsWith('.json')),
    presets: presets.presets,
  }
}

export function uploadVoice(user: string, file: File) {
  const body = new FormData()
  body.append('file', file)
  return call(`/reference-audio/upload${forUser(user)}`, { method: 'POST', body })
}

export function deleteVoice(user: string, filename: string) {
  return call(`/reference-audio/${encodeURIComponent(filename)}${forUser(user)}`, { method: 'DELETE' })
}

function toDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('Could not read the audio'))
    reader.readAsDataURL(blob)
  })
}

export async function saveDesignedVoice(user: string, filename: string, description: string, audio: Blob) {
  const audio_b64 = await toDataUrl(audio)
  return call(`/reference-audio/save-design${forUser(user)}`, jsonBody({ filename, description, audio_b64 }))
}

export function savePreset(user: string, preset: Preset) {
  return call(`/voice-presets${forUser(user)}`, jsonBody(preset))
}

export function deletePreset(user: string, name: string) {
  return call(`/voice-presets/${encodeURIComponent(name)}${forUser(user)}`, { method: 'DELETE' })
}

// Generates through the progress stream and resolves with the finished WAV.
export async function synthesize(
  request: TtsRequest,
  onProgress: (progress: Progress) => void,
  signal: AbortSignal,
): Promise<Blob> {
  const response = await fetch(`${BASE}/api/tts/stream`, { ...jsonBody(request), signal })
  if (!response.ok) throw new Error(await errorMessage(response))

  const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader()
  const parser = new SseParser()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) throw new Error('The connection closed before the audio was finished')

    for (const { event, data } of parser.feed(value)) {
      const payload = JSON.parse(data)
      if (event === 'error') throw new Error(payload.detail ?? 'Generation failed')
      if (event === 'progress') {
        onProgress({ percent: payload.overall_pct, chunk: payload.chunk, of: payload.of })
      }
      if (event === 'done') {
        const bytes = Uint8Array.from(atob(payload.audio_b64), (char) => char.charCodeAt(0))
        return new Blob([bytes], { type: 'audio/wav' })
      }
    }
  }
}
