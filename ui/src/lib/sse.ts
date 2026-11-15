export interface SseEvent {
  event: string
  data: string
}

// Splits a text/event-stream into events. Feed it chunks as they arrive; a chunk can end anywhere.
export class SseParser {
  private buffer = ''

  feed(chunk: string): SseEvent[] {
    this.buffer += chunk
    const blocks = this.buffer.split('\n\n')
    this.buffer = blocks.pop() ?? ''
    return blocks.flatMap((block) => {
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      return data.length ? [{ event, data: data.join('\n') }] : []
    })
  }
}
