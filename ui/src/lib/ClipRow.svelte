<script lang="ts">
  import { defaultVoiceConflict, saveDesignedVoice, savePreset, type Voices } from './api'

  export interface Clip {
    id: number
    url: string
    audio: Blob
    text: string
    voice: string
    // Set when the clip came from a typed-in voice description.
    design?: string
  }

  interface Props {
    clip: Clip
    user: string
    voices: Voices
    onremove: () => void
    onsaved: () => void
  }

  let { clip, user, voices, onremove, onsaved }: Props = $props()

  let name = $state('')
  let error = $state('')
  let saved = $state('')
  let saving = $state(false)

  const filename = $derived.by(() => {
    const trimmed = name.trim()
    return /\.\w+$/.test(trimmed) ? trimmed : `${trimmed}.wav`
  })
  const download = $derived(
    `${clip.voice} ${clip.text.slice(0, 30)}`.replace(/[^\w -]+/g, '').trim().replace(/\s+/g, '-') + '.wav',
  )

  async function save(action: () => Promise<unknown>, label: string) {
    saving = true
    error = saved = ''
    try {
      await action()
      saved = `Saved ${label}`
      name = ''
      onsaved()
    } catch (e) {
      error = (e as Error).message
    } finally {
      saving = false
    }
  }

  function saveClone() {
    if (name.includes('/')) return void (error = "Names can't contain /")
    const conflict = defaultVoiceConflict(user, filename, voices, true)
    if (conflict) return void (error = `${filename} conflicts with default voice ${conflict}`)
    if (voices.files.some((file) => file.filename === filename)) return void (error = `${filename} already exists`)
    return save(() => saveDesignedVoice(user, filename, clip.design ?? '', clip.audio), filename)
  }

  function savePresetFromClip() {
    const presetName = name.trim()
    const conflict = defaultVoiceConflict(user, presetName, voices, false)
    if (conflict) return void (error = `${presetName} conflicts with default voice ${conflict}`)
    if (voices.presets.some((preset) => preset.name === presetName)) return void (error = `${presetName} already exists`)
    return save(() => savePreset(user, { name: presetName, voice_description: clip.design ?? '' }), presetName)
  }
</script>

<li class="clip">
  <p class="clip-head"><strong>{clip.voice}</strong> <span class="dim clip-text">{clip.text}</span></p>
  <audio controls src={clip.url}></audio>
  <div class="actions">
    <a class="button" href={clip.url} download={download}>Download</a>
    <button type="button" onclick={onremove}>Remove</button>
  </div>
  {#if clip.design}
    <div class="keep">
      <label>
        Keep this voice
        <input bind:value={name} placeholder="Name" maxlength="60" />
      </label>
      <div class="actions">
        <button type="button" disabled={!name.trim() || saving} onclick={saveClone}>Save as clone</button>
        <button type="button" disabled={!name.trim() || saving} onclick={savePresetFromClip}>Save as preset</button>
      </div>
      <p class="dim hint">A clone reuses this exact sound. A preset re-imagines the description each time.</p>
    </div>
  {/if}
  {#if saved}<p role="status">{saved}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</li>
