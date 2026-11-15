import assert from 'node:assert/strict'
import { test } from 'node:test'
import { SseParser } from './sse.ts'

test('parses complete events', () => {
  const events = new SseParser().feed('event: progress\ndata: {"a":1}\n\nevent: done\ndata: {}\n\n')
  assert.deepEqual(events, [
    { event: 'progress', data: '{"a":1}' },
    { event: 'done', data: '{}' },
  ])
})

test('holds a partial event until the rest arrives', () => {
  const parser = new SseParser()
  assert.deepEqual(parser.feed('event: do'), [])
  assert.deepEqual(parser.feed('ne\ndata: x'), [])
  assert.deepEqual(parser.feed('yz\n\n'), [{ event: 'done', data: 'xyz' }])
})

test('ignores blocks with no data', () => {
  assert.deepEqual(new SseParser().feed(': keepalive\n\n'), [])
})
