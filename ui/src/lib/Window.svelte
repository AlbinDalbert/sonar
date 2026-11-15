<script lang="ts">
  import type { Snippet } from 'svelte'

  let { title, children }: { title: string; children: Snippet } = $props()
</script>

<section class="win">
  <h2 class="bar">{title}</h2>
  <div class="body">{@render children()}</div>
</section>
