<script lang="ts">
  import {
    defaultVoiceConflict,
    deletePreset,
    deleteVoice,
    fileLabel,
    isInheritedVoice,
    savePreset,
    uploadVoice,
    type Preset,
    type Voices,
  } from './api'
  import Window from './Window.svelte'

  interface Props {
    user: string
    voices: Voices
    voice: string
    onchange: () => Promise<void>
  }

  let { user, voices, voice = $bindable(), onchange }: Props = $props()

  let error = $state('')
  let working = $state(false)
  let confirming = $state('')
  let fileInput = $state<HTMLInputElement>()

  let editing = $state(false)
  let isNew = $state(true)
  let form = $state({ name: '', voice_description: '', description: '' })

  $effect(() => {
    void user
    editing = false
    confirming = ''
    error = ''
  })

  async function run(action: () => Promise<unknown>) {
    working = true
    error = ''
    try {
      await action()
      await onchange()
      return true
    } catch (e) {
      error = (e as Error).message
      return false
    } finally {
      working = false
    }
  }

  async function upload() {
    const file = fileInput?.files?.[0]
    if (fileInput) fileInput.value = ''
    if (!file) return
    const conflict = defaultVoiceConflict(user, file.name, voices, true)
    if (conflict) {
      error = `${file.name} conflicts with default voice ${conflict}`
      return
    }
    const existing = voices.files.some((item) => !isInheritedVoice(user, item) && item.filename === file.name)
    if (existing && !confirm(`Replace ${file.name}?`)) return
    await run(() => uploadVoice(user, file))
  }

  // First press arms the button, second press deletes.
  function remove(key: string, action: () => Promise<unknown>) {
    if (confirming !== key) return void (confirming = key)
    confirming = ''
    run(action)
  }

  function edit(preset?: Preset) {
    isNew = !preset
    form = {
      name: preset?.name ?? '',
      voice_description: preset?.voice_description ?? '',
      description: preset?.description ?? '',
    }
    editing = true
  }

  async function submit(event: SubmitEvent) {
    event.preventDefault()
    const name = form.name.trim()
    const conflict = defaultVoiceConflict(user, name, voices, false)
    if (conflict) {
      error = `${name} conflicts with default voice ${conflict}`
      return
    }
    if (isNew && voices.presets.some((preset) => preset.name === name)) {
      error = `${name} already exists`
      return
    }
    const saved = await run(() =>
      savePreset(user, { name, voice_description: form.voice_description.trim(), description: form.description.trim() }),
    )
    if (saved) editing = false
  }
</script>

{#snippet deleteButton(key: string, label: string, action: () => Promise<unknown>)}
  <button
    type="button"
    class="small"
    class:armed={confirming === key}
    aria-label={confirming === key ? `Confirm deleting ${label}` : `Delete ${label}`}
    disabled={working}
    onclick={() => remove(key, action)}
    onblur={() => confirming === key && (confirming = '')}
  >
    {confirming === key ? 'Sure?' : 'Delete'}
  </button>
{/snippet}

<Window title="Voices">
  {#if error}<p class="error" role="alert">{error}</p>{/if}

  <div class="group-head">
    <h3>Clones</h3>
    <button type="button" class="small" disabled={working} onclick={() => fileInput?.click()}>Upload</button>
    <input bind:this={fileInput} type="file" accept="audio/*" hidden onchange={upload} />
  </div>
  {#if voices.files.length}
    <ul class="voices">
      {#each voices.files as file (file.filename)}
        {@const value = `clone:${file.filename}`}
        {@const shared = isInheritedVoice(user, file)}
        <li class:selected={voice === value}>
          <button type="button" class="pick" aria-pressed={voice === value} onclick={() => (voice = value)}>
            <span>{fileLabel(file.filename, voices.files)}</span>
            {#if file.description}<span class="dim desc">{file.description}</span>{/if}
          </button>
          {#if shared}
            <span class="dim tag">default</span>
          {:else}
            {@render deleteButton(value, file.filename, () => deleteVoice(user, file.filename))}
          {/if}
        </li>
      {/each}
    </ul>
  {:else}
    <p class="dim hint">No cloned voices for {user} yet. Upload a short clip of someone speaking.</p>
  {/if}

  <div class="group-head">
    <h3>Presets</h3>
    <button type="button" class="small" onclick={() => edit()}>New</button>
  </div>

  {#if editing}
    <form class="form preset-form" onsubmit={submit}>
      <label>Name <input bind:value={form.name} disabled={!isNew} required maxlength="60" /></label>
      <label>
        Voice description
        <textarea bind:value={form.voice_description} rows="4" required></textarea>
      </label>
      <label>Note <input bind:value={form.description} /></label>
      <div class="actions">
        <button class="primary" type="submit" disabled={working}>Save</button>
        <button type="button" onclick={() => (editing = false)}>Cancel</button>
      </div>
    </form>
  {/if}

  {#if voices.presets.length}
    <ul class="voices">
      {#each voices.presets as preset (preset.name)}
        {@const value = `preset:${preset.name}`}
        {@const shared = isInheritedVoice(user, preset)}
        <li class:selected={voice === value}>
          <button type="button" class="pick" aria-pressed={voice === value} onclick={() => (voice = value)}>
            <span>{preset.name}</span>
            <span class="dim desc">{preset.description || preset.voice_description}</span>
          </button>
          {#if shared}
            <span class="dim tag">default</span>
          {:else}
            <button type="button" class="small" aria-label={`Edit ${preset.name}`} onclick={() => edit(preset)}>Edit</button>
            {@render deleteButton(value, preset.name, () => deletePreset(user, preset.name))}
          {/if}
        </li>
      {/each}
    </ul>
  {:else if !editing}
    <p class="dim hint">No presets for {user} yet.</p>
  {/if}
</Window>
