# sonar ui

Web UI for the [Sonar API](../api): generate speech, manage cloned voices and presets, and design
new voices. Add `?user=name` to the URL to work in that user's voice namespace.

Svelte 5 + TypeScript + Vite, built to static files. nginx serves them and proxies `/tts-api` to
the API, so no CORS is needed.

## Development

```sh
npm install
npm run dev      # http://localhost:5173, proxies /tts-api to http://localhost:8888
```

Set `TTS_API_URL` to use another API. `npm test` runs the unit tests and `npm run check`
type-checks.

## Image

Unprivileged nginx (uid 101) on port **8080**. One setting:

| Variable      | Default                 | |
|---------------|-------------------------|-|
| `TTS_API_URL` | `http://sonar-api:8888` | The API's address from the container. No trailing slash. It must resolve when nginx starts. |

`/healthz` answers 200 for probes. Generated audio streams through unbuffered.

```sh
docker run -p 8080:8080 -e TTS_API_URL=http://host.docker.internal:8888 ghcr.io/albindalbert/sonar-ui:main
```
