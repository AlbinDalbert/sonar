// Plays audio that arrives in pieces, back to back, through Web Audio.
//
// Create it from a click or key handler: browsers only let an AudioContext start from a user gesture.
export class StreamPlayer {
  private context = new AudioContext()
  private nextTime = 0
  // How far ahead of the playhead new audio is queued. It grows when generation can't keep up, so a
  // slow stream stutters once or twice and then plays smoothly.
  private lead = 0.2
  private sources = new Set<AudioBufferSourceNode>()

  constructor() {
    void this.context.resume()
  }

  play(samples: Float32Array, sampleRate: number) {
    if (!samples.length || this.context.state === 'closed') return
    const buffer = this.context.createBuffer(1, samples.length, sampleRate)
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0)

    const now = this.context.currentTime
    if (this.nextTime < now) {
      if (this.nextTime > 0) this.lead = Math.min(this.lead + 0.5, 3)
      this.nextTime = now + this.lead
    }

    const source = this.context.createBufferSource()
    source.buffer = buffer
    source.connect(this.context.destination)
    source.onended = () => this.sources.delete(source)
    source.start(this.nextTime)
    this.sources.add(source)
    this.nextTime += buffer.duration
  }

  // Resolves once everything queued so far has played.
  finished(): Promise<void> {
    const remaining = this.nextTime - this.context.currentTime
    return new Promise((resolve) => setTimeout(resolve, Math.max(0, remaining) * 1000))
  }

  stop() {
    this.sources.forEach((source) => source.stop())
    this.sources.clear()
    void this.context.close()
  }
}
