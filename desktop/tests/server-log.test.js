// node --test desktop/tests
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { createServerLog } = require('../server-log')

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'mc-log-'))
}

test('appends every chunk to the log', () => {
  const dir = tempDir()
  const log = createServerLog(path.join(dir, 'server.log'), path.join(dir, 'fallback.log'))
  log.write('one\n')
  log.write('two\n')
  assert.equal(fs.readFileSync(path.join(dir, 'server.log'), 'utf8'), 'one\ntwo\n')
})

test('keeps the file closed between writes, so nothing is locked or lost to one bad open', () => {
  const dir = tempDir()
  const file = path.join(dir, 'server.log')
  const log = createServerLog(file, null)
  log.write('before\n')
  fs.renameSync(file, path.join(dir, 'moved.log')) // impossible on Windows while held open
  log.write('after\n')
  assert.equal(fs.readFileSync(file, 'utf8'), 'after\n')
})

test('a log that cannot be written moves to the fallback and says why', () => {
  const dir = tempDir()
  const blocker = path.join(dir, 'not-a-folder')
  fs.writeFileSync(blocker, 'a file where the log folder should be')
  const fallback = path.join(dir, 'workspace', 'server.log')
  const log = createServerLog(path.join(blocker, 'server.log'), fallback)

  assert.equal(log.write('server output\n'), true)
  const text = fs.readFileSync(fallback, 'utf8')
  assert.match(text, /could not write .*server\.log/)
  assert.match(text, /server output/)
  assert.equal(log.path, fallback)
})

test('when nothing can be written it retries later instead of going quiet for good', () => {
  const dir = tempDir()
  const blocker = path.join(dir, 'blocker')
  fs.writeFileSync(blocker, 'x')
  const target = path.join(blocker, 'server.log')
  const log = createServerLog(target, null)
  assert.equal(log.write('lost\n'), false)

  fs.rmSync(blocker) // whatever was in the way is gone
  assert.equal(log.write('back\n'), true)
  assert.equal(fs.readFileSync(target, 'utf8'), 'back\n')
})

test('rotates instead of growing forever', () => {
  const dir = tempDir()
  const file = path.join(dir, 'server.log')
  const log = createServerLog(file, null, { maxBytes: 20 })
  log.write('0123456789\n') // 11
  log.write('abcdefghij\n') // 22 > 20: rotate first
  assert.equal(fs.readFileSync(`${file}.1`, 'utf8'), '0123456789\n')
  assert.equal(fs.readFileSync(file, 'utf8'), 'abcdefghij\n')
})
