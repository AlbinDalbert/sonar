# Deploying to Kubernetes

Two [kustomize](https://kustomize.io) bases, with nothing cluster-specific in them:

- [`api/`](api): the API Deployment (one NVIDIA GPU, `runtimeClassName: nvidia`), its Service
  `sonar-api:8888`, and a 5 GiB PVC for voices. Models come from a hostPath Hugging Face cache at
  `/var/lib/sonar-models`; the image runs offline, so download the models there first.
- [`ui/`](ui): the UI Deployment and Service `sonar-ui:80`. It reaches the API at
  `http://sonar-api:8888`, i.e. in the same namespace.

Both images default to `:main`. If the GHCR packages are private, add an `imagePullSecrets` patch.

Namespace, node placement, ingress and settings belong in an overlay in your own infrastructure
repo.

## An overlay

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: sonar

resources:
  # Pin `ref` to a tag or commit to control when manifest changes reach you.
  - https://github.com/AlbinDalbert/sonar//deploy/api?ref=main
  - https://github.com/AlbinDalbert/sonar//deploy/ui?ref=main
  - ingress.yaml

images:
  - name: ghcr.io/albindalbert/sonar-api
    newTag: main # or sha-1a2b3c4, or a release
  - name: ghcr.io/albindalbert/sonar-ui
    newTag: main

configMapGenerator:
  - name: sonar-api-config
    envs: [config.env]

patches:
  - target: { kind: Deployment, name: sonar-api }
    patch: |-
      - op: add
        path: /spec/template/spec/nodeSelector
        value: { kubernetes.io/hostname: my-gpu-node }
```

`config.env` holds API settings (every variable is in [`api/README.md`](../api/README.md)). For a
small (~3 GiB) GPU, keep the auxiliary modules on CPU:

```sh
QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
QWEN_MAX_CHUNK_CHARS=220
QWEN_MAX_NEW_TOKENS=192
QWEN_MAX_NEW_TOKENS_PER_CHAR=1.0
QWEN_MAX_NEW_TOKENS_BUFFER=48
QWEN_SPEECH_TOKENIZER_DEVICE=cpu
QWEN_SPEAKER_ENCODER_DEVICE=cpu
```

For ~8 GiB, keep them on the GPU for lower latency:

```sh
QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
QWEN_MAX_CHUNK_CHARS=250
QWEN_MAX_NEW_TOKENS=512
QWEN_MAX_NEW_TOKENS_PER_CHAR=2.0
QWEN_MAX_NEW_TOKENS_BUFFER=64
QWEN_SPEECH_TOKENIZER_DEVICE=cuda:0
QWEN_SPEAKER_ENCODER_DEVICE=cuda:0
```

`ingress.yaml` points a host at `sonar-ui` port `http`. Expose `sonar-api` too only if other
clients call the API directly.

The UI and API can also live in different namespaces or clusters: deploy each base on its own and
set the UI's `TTS_API_URL` to wherever the API is (e.g. `http://sonar-api.other-ns.svc:8888`).

### Keeping existing volumes

To reuse a voices PVC and model cache that already exist under other names, drop the base PVC and
point the Deployment at them:

```yaml
patches:
  - patch: |-
      $patch: delete
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: sonar-reference-audio
  - target: { kind: Deployment, name: sonar-api }
    patch: |-
      - op: replace
        path: /spec/template/spec/volumes/0/hostPath/path
        value: /var/lib/old-models-path
      - op: replace
        path: /spec/template/spec/volumes/1/persistentVolumeClaim/claimName
        value: old-reference-audio-claim
```

## Updating

The Deployments use `imagePullPolicy: Always`, so `kubectl rollout restart deploy/sonar-api`
(or `sonar-ui`) pulls the latest `:main`. Or pin `newTag` to a `sha-<commit>` and apply again.

With a single GPU, the API's new pod can only start once the old one has released the GPU.
