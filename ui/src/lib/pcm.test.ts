import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Pcm16Decoder, wavFromPcm } from './pcm.ts'

test('decodes little-endian samples', () => {
  const samples = new Pcm16Decoder().feed(new Uint8Array([0x00, 0x40, 0x00, 0x80, 0xff, 0x7f]))
  assert.deepEqual([...samples], [0.5, -1, 32767 / 32768])
})

test('carries a split sample over to the next chunk', () => {
  const decoder = new Pcm16Decoder()
  assert.deepEqual([...decoder.feed(new Uint8Array([0x00, 0x40, 0x00]))], [0.5])
  assert.deepEqual([...decoder.feed(new Uint8Array([0xc0]))], [-0.5])
})

test('builds a WAV with the real sizes', async () => {
  const blob = wavFromPcm([new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])], 24000)
  const bytes = new Uint8Array(await blob.arrayBuffer())
  const view = new DataView(bytes.buffer)
  assert.equal(bytes.length, 44 + 4)
  assert.equal(new TextDecoder().decode(bytes.subarray(0, 4)), 'RIFF')
  assert.equal(view.getUint32(4, true), 36 + 4)
  assert.equal(view.getUint32(24, true), 24000)
  assert.equal(view.getUint32(40, true), 4)
  assert.deepEqual([...bytes.subarray(44)], [1, 2, 3, 4])
})
