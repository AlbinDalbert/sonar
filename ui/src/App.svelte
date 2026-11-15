<script lang="ts">
  import { onMount } from 'svelte'
  import { getInfo, getStatus, listVoices, type Status, type Voices } from './lib/api'
  import Mark from './lib/Mark.svelte'
  import Speak from './lib/Speak.svelte'
  import VoiceManager from './lib/VoiceManager.svelte'

  const FALLBACK_LANGUAGES = ['Auto', 'English']

  const initialUser = new URLSearchParams(location.search).get('user')?.trim() || 'default'
  let user = $state(initialUser)
  let userInput = $state(initialUser)
  let userError = $state('')

  let voices = $state.raw<Voices>({ files: [], presets: [] })
  let voice = $state('')
  let loadError = $state('')
  let languages = $state.raw(FALLBACK_LANGUAGES)
  let status = $state.raw<Status | null>(null)
  let offline = $state(false)

  async function loadVoices() {
    const requested = user
    try {
      const loaded = await listVoices(requested)
      if (requested !== user) return
      voices = loaded
      loadError = ''
      const options = [
        ...loaded.files.map((file) => `clone:${file.filename}`),
        ...loaded.presets.map((preset) => `preset:${preset.name}`),
        'design',
      ]
      if (!options.includes(voice)) voice = options[0]
    } catch (e) {
      loadError = (e as Error).message
    }
  }

  function changeUser() {
    const next = userInput.trim() || 'default'
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(next)) {
      userError = 'Letters, numbers, - and _ only, up to 32'
      return
    }
    userError = ''
    userInput = next
    if (next === user) return
    user = next
    const url = new URL(location.href)
    if (next === 'default') url.searchParams.delete('user')
    else url.searchParams.set('user', next)
    history.replaceState(null, '', url)
    loadVoices()
  }

  async function pollStatus() {
    if (document.hidden) return
    try {
      status = await getStatus()
      offline = false
    } catch {
      offline = true
    }
  }

  onMount(() => {
    loadVoices()
    pollStatus()
    getInfo()
      .then((info) => (languages = info.supported_languages))
      .catch(() => {})
    const timer = setInterval(pollStatus, 5000)
    return () => clearInterval(timer)
  })

  const statusText = $derived.by(() => {
    if (offline) return 'API offline'
    if (!status) return 'Connecting'
    const model = status.model_id ? status.model_id.split('/').pop() : status.model_state
    const memory = status.cuda ? ` · ${Math.round(status.cuda.free_mb / 1024 * 10) / 10} GB free` : ''
    return `${model}${memory}`
  })
</script>

<div class="page">
  <header>
    <h1><Mark size={18} />Sonar</h1>
    <div class="header-side">
      <p class="dim status-line" class:error={offline} role="status">{statusText}</p>
      <label class="user">
        User
        <input
          bind:value={userInput}
          onchange={changeUser}
          autocomplete="off"
          autocapitalize="off"
          spellcheck="false"
          maxlength="32"
        />
      </label>
      {#if userError}<p class="error" role="alert">{userError}</p>{/if}
    </div>
  </header>

  {#if loadError}<p class="error" role="alert">Couldn't load voices: {loadError}</p>{/if}

  <main>
    <Speak {user} {voices} {languages} bind:voice onsaved={loadVoices} />
    <VoiceManager {user} {voices} bind:voice onchange={loadVoices} />
  </main>
</div>
