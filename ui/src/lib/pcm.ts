// Raw 16-bit little-endian mono PCM, as /api/tts/audio-stream sends it with audio_format "pcm".

// Turns network chunks into float samples. A chunk can end in the middle of a sample, so the odd
// byte is carried over to the next one.
export class Pcm16Decoder {
  private carry: number | null = null

  feed(chunk: Uint8Array): Float32Array {
    let bytes = chunk
    if (this.carry !== null) {
      bytes = new Uint8Array(chunk.length + 1)
      bytes[0] = this.carry
      bytes.set(chunk, 1)
      this.carry = null
    }
    const count = bytes.length >> 1
    if (bytes.length & 1) this.carry = bytes[bytes.length - 1]

    const view = new DataView(bytes.buffer, bytes.byteOffset, count * 2)
    const samples = new Float32Array(count)
    for (let i = 0; i < count; i++) samples[i] = view.getInt16(i * 2, true) / 32768
    return samples
  }
}

// Wraps the PCM chunks in a WAV header with the real sizes.
export function wavFromPcm(chunks: Uint8Array[], sampleRate: number): Blob {
  let length = chunks.reduce((total, chunk) => total + chunk.length, 0)
  length -= length & 1

  const header = new DataView(new ArrayBuffer(44))
  const text = (offset: number, value: string) =>
    [...value].forEach((char, i) => header.setUint8(offset + i, char.charCodeAt(0)))
  text(0, 'RIFF')
  header.setUint32(4, 36 + length, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  header.setUint32(16, 16, true) // fmt chunk size
  header.setUint16(20, 1, true) // PCM
  header.setUint16(22, 1, true) // mono
  header.setUint32(24, sampleRate, true)
  header.setUint32(28, sampleRate * 2, true) // byte rate
  header.setUint16(32, 2, true) // block align
  header.setUint16(34, 16, true) // bits per sample
  text(36, 'data')
  header.setUint32(40, length, true)

  const data = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    const part = chunk.subarray(0, length - offset)
    data.set(part, offset)
    offset += part.length
  }
  return new Blob([header.buffer, data], { type: 'audio/wav' })
}
