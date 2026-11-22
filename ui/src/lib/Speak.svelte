<script lang="ts">
  import { onDestroy } from 'svelte'
  import { fileLabel, streamSpeech, type TtsRequest, type Voices } from './api'
  import ClipRow, { type Clip } from './ClipRow.svelte'
  import { StreamPlayer } from './player'
  import Window from './Window.svelte'

  interface Props {
    user: string
    voices: Voices
    languages: string[]
    voice: string
    onsaved: () => void
  }

  let { user, voices, languages, voice = $bindable(), onsaved }: Props = $props()

  let text = $state('')
  let description = $state('')
  let language = $state('Auto')
  let modelSize = $state('fast')
  let cpuMode = $state(false)

  let busy = $state(false)
  // Live playback can outlast the request: the end of the audio arrives faster than it plays.
  let playing = $state(false)
  let seconds = $state(0)
  let error = $state('')
  let notice = $state('')
  let clips = $state.raw<Clip[]>([])

  let controller: AbortController | null = null
  let player: StreamPlayer | null = null
  let nextId = 1

  const designing = $derived(voice === 'design' || voice.startsWith('preset:'))
  const ready = $derived(!!text.trim() && !!voice && (voice !== 'design' || !!description.trim()))
  const voiceName = (value: string) => value.slice(value.indexOf(':') + 1).replace(/\.\w+$/, '')

  function buildRequest(): TtsRequest {
    const base = { text: text.trim(), user, language, cpu_mode: cpuMode, audio_format: 'pcm' as const }
    if (voice === 'design') {
      return { ...base, mode: 'design', model_size: 'quality', voice_description: description.trim() }
    }
    if (voice.startsWith('preset:')) {
      return { ...base, mode: 'design', model_size: 'quality', voice_preset: voice.slice('preset:'.length) }
    }
    return { ...base, mode: 'clone', model_size: modelSize, reference_audio: voice.slice('clone:'.length) }
  }

  async function generate() {
    if (!ready || busy) return
    const request = buildRequest()
    const design = voice === 'design' ? description.trim() : undefined
    const label = voice === 'design' ? 'New voice' : voiceName(voice)

    stopPlayback()
    // Made here, still inside the click or key press, so the browser lets it play.
    const live = new StreamPlayer()
    player = live
    playing = true
    controller = new AbortController()
    busy = true
    error = notice = ''
    seconds = 0
    let complete = false
    try {
      const { audio, stopped } = await streamSpeech(
        request,
        (samples, sampleRate) => {
          live.play(samples, sampleRate)
          seconds += samples.length / sampleRate
        },
        controller.signal,
      )
      const clip = { id: nextId++, url: URL.createObjectURL(audio), audio, text: request.text, voice: label, design }
      clips = [clip, ...clips]
      if (stopped) notice = 'Stopped. The clip has what was generated up to then.'
      complete = !stopped
    } catch (e) {
      if ((e as Error).name === 'AbortError') notice = 'Stopped before any audio arrived.'
      else error = (e as Error).message
    } finally {
      busy = false
      controller = null
    }
    if (!complete) return stopPlayback(live)
    await live.finished()
    stopPlayback(live)
  }

  // Stops the given player, or whichever is playing, unless a newer one has taken over.
  function stopPlayback(which: StreamPlayer | null = player) {
    if (!which || which !== player) return
    which.stop()
    player = null
    playing = false
  }

  function stop() {
    controller?.abort()
    stopPlayback()
  }

  function remove(clip: Clip) {
    URL.revokeObjectURL(clip.url)
    clips = clips.filter((other) => other !== clip)
  }

  function onkeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) generate()
  }

  onDestroy(() => {
    stop()
    clips.forEach((clip) => URL.revokeObjectURL(clip.url))
  })
</script>

<div class="stack">
  <Window title="Speak">
    <form
      class="form"
      onsubmit={(event) => {
        event.preventDefault()
        generate()
      }}
    >
      <label>
        <span class="label-row">Text <span>{text.length} characters</span></span>
        <textarea class="prose" bind:value={text} {onkeydown} rows="8" placeholder="Type something to say"></textarea>
      </label>

      <label>
        Voice
        <select bind:value={voice}>
          {#if voices.files.length}
            <optgroup label="Clones">
              {#each voices.files as file (file.filename)}
                <option value={`clone:${file.filename}`}>{fileLabel(file.filename, voices.files)}</option>
              {/each}
            </optgroup>
          {/if}
          {#if voices.presets.length}
            <optgroup label="Presets">
              {#each voices.presets as preset (preset.name)}
                <option value={`preset:${preset.name}`}>{preset.name}</option>
              {/each}
            </optgroup>
          {/if}
          <option value="design">Describe a new voice…</option>
        </select>
      </label>

      {#if voice === 'design'}
        <label>
          Voice description
          <textarea
            bind:value={description}
            {onkeydown}
            rows="3"
            placeholder="Deep, calm older voice with a slight rasp"
          ></textarea>
        </label>
      {/if}

      <div class="row">
        <label>
          Language
          <select bind:value={language}>
            {#each languages as option (option)}<option>{option}</option>{/each}
          </select>
        </label>
        <label>
          Model
          {#if designing}
            <select disabled><option>Quality</option></select>
          {:else}
            <select bind:value={modelSize}>
              <option value="fast">Fast</option>
              <option value="quality">Quality</option>
            </select>
          {/if}
        </label>
      </div>

      <label class="check">
        <input type="checkbox" bind:checked={cpuMode} />
        Use CPU instead of GPU <span class="dim">(much slower)</span>
      </label>

      <div class="actions">
        {#if busy || playing}
          <button type="button" onclick={stop}>Stop</button>
        {:else}
          <button class="primary" type="submit" disabled={!ready}>Generate</button>
        {/if}
      </div>

      {#if busy || playing}
        <p class="dim hint" role="status">
          {#if busy && !seconds}
            Loading the model. This takes a few seconds.
          {:else if busy}
            Playing as it generates, {seconds.toFixed(1)} s so far
          {:else}
            Generated {seconds.toFixed(1)} s, finishing playback
          {/if}
        </p>
      {/if}
      {#if notice}<p class="dim hint" role="status">{notice}</p>{/if}
      {#if error}<p class="error" role="alert">{error}</p>{/if}
    </form>
  </Window>

  {#if clips.length}
    <Window title="Clips">
      <ul class="clips">
        {#each clips as clip (clip.id)}
          <ClipRow
            {clip}
            {user}
            takenFiles={voices.files.map((file) => file.filename)}
            takenPresets={voices.presets.map((preset) => preset.name)}
            onremove={() => remove(clip)}
            {onsaved}
          />
        {/each}
      </ul>
    </Window>
  {/if}
</div>
