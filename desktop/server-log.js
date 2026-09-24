// The server's output, kept in a file people (and support) can read.
//
// This used to be one fs.createWriteStream opened at start-up, with no
// 'error' listener. When that open failed, every later write was dropped
// without a trace: an installed app ran for over a week of calls with a
// server.log last written on 15 September, and a silent in-call agent could
// not be diagnosed because nothing it did was recorded.
//
// Now each chunk is appended on its own (the file is not held open, so a
// transient lock, an antivirus scan or the folder being renamed costs one
// line, not the rest of the session), a failure moves to a fallback file
// instead of giving up, and the file is rotated so it cannot grow forever.

const fs = require('node:fs')
const path = require('node:path')

const MAX_BYTES = 5 * 1024 * 1024

function createServerLog(primaryPath, fallbackPath, { maxBytes = MAX_BYTES } = {}) {
  const candidates = [primaryPath, fallbackPath].filter(Boolean)
  let index = 0
  let written = null // bytes in the current file, learned on first write

  function rotateIfFull(file, incoming) {
    if (written === null) {
      try {
        written = fs.statSync(file).size
      } catch {
        written = 0
      }
    }
    if (written + incoming <= maxBytes) return
    try {
      fs.renameSync(file, `${file}.1`) // replaces the previous .1
    } catch {
      // Could not rotate; keep appending rather than lose output.
    }
    written = 0
  }

  function write(text) {
    const data = String(text)
    while (index < candidates.length) {
      const file = candidates[index]
      try {
        fs.mkdirSync(path.dirname(file), { recursive: true })
        rotateIfFull(file, Buffer.byteLength(data))
        fs.appendFileSync(file, data)
        written += Buffer.byteLength(data)
        return true
      } catch (error) {
        index += 1
        written = null
        const next = candidates[index]
        if (next) {
          try {
            fs.mkdirSync(path.dirname(next), { recursive: true })
            fs.appendFileSync(next, `[meet-companion] could not write ${file} (${error.code || error.message}); logging here instead\n`)
          } catch {
            // The loop tries it properly next.
          }
        }
      }
    }
    // Nowhere worked this time: start from the top on the next chunk
    // rather than going quiet for the rest of the session.
    index = 0
    written = null
    return false
  }

  return {
    write,
    get path() {
      return candidates[Math.min(index, candidates.length - 1)]
    },
  }
}

module.exports = { createServerLog, MAX_BYTES }
