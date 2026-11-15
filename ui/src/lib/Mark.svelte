<!-- A ping: a disc split by two wavefronts spreading from a source at its right edge. The piece by
     the source is the voice, in the lighter amber; the two after it are the sound it sends out.
     Plain paths (no masks or ids), so any number of marks can share a page. -->
<script lang="ts">
  let { size = 16 }: { size?: number } = $props()
</script>

<svg class="mark" viewBox="0 0 16 16" width={size} height={size} aria-hidden="true">
  <path class="source" d="M14.19 3.76A4.25 4.25 0 0 0 14.19 12.24A7.5 7.5 0 0 0 14.19 3.76z" />
  <path class="waves" d="M9.55 0.66A8.85 8.85 0 0 0 9.55 15.34A7.5 7.5 0 0 0 13.03 13.56A5.75 5.75 0 0 1 13.03 2.44A7.5 7.5 0 0 0 9.55 0.66zM7.34 0.53A7.5 7.5 0 0 0 7.34 15.47A10.35 10.35 0 0 1 7.34 0.53z" />
</svg>

<style>
  .mark {
    display: block;
    flex: none;
  }
  .source {
    fill: var(--accent-strong);
  }
  .waves {
    fill: var(--accent);
  }
</style>
