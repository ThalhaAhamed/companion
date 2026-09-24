// Meet Companion desktop shell.
//
// The app is the same web UI and Python server as the self-hosted version.
// This shell starts the bundled server on a free localhost port, waits for
// it to answer, and opens it in a window. All data lives in the user's app
// data directory; the installed files are never written to.

const { app, BrowserWindow, dialog, session, shell } = require('electron')
const { spawn } = require('node:child_process')
const http = require('node:http')
const path = require('node:path')
const fs = require('node:fs')
const { createServerLog } = require('./server-log')

let serverProcess = null
let serverPort = null
let mainWindow = null

function serverExecutable() {
  const name = process.platform === 'win32' ? 'meet-companion-server.exe' : 'meet-companion-server'
  return app.isPackaged
    ? path.join(process.resourcesPath, 'server', name)
    : path.join(__dirname, 'build', 'meet-companion-server', name)
}

function dataDir() {
  // Overridable so a developer can run the shell against an existing
  // workspace (for example the repository checkout) instead of a fresh one.
  return process.env.MEET_COMPANION_DATA_DIR || path.join(app.getPath('userData'), 'workspace')
}

function waitForHealth(port, timeoutMs) {
  const started = Date.now()
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get({ host: '127.0.0.1', port, path: '/health', timeout: 1500 }, (response) => {
        response.resume()
        if (response.statusCode === 200) return resolve()
        retry()
      })
      request.on('error', retry)
      request.on('timeout', () => {
        request.destroy()
        retry()
      })
    }
    const retry = () => {
      if (serverProcess && serverProcess.exitCode !== null) {
        return reject(new Error(`The server exited with code ${serverProcess.exitCode}.`))
      }
      if (Date.now() - started > timeoutMs) return reject(new Error('The server did not start in time.'))
      setTimeout(attempt, 300)
    }
    attempt()
  })
}

function startServer() {
  const executable = serverExecutable()
  if (!fs.existsSync(executable)) {
    throw new Error(`Server binary not found at ${executable}. Build it with "pyinstaller desktop/server.spec" first.`)
  }
  const dir = dataDir()
  fs.mkdirSync(dir, { recursive: true })

  // Falls back to the data directory if the usual file cannot be written.
  const log = createServerLog(path.join(app.getPath('userData'), 'server.log'), path.join(dir, 'server.log'))
  log.write(`
[meet-companion] starting ${app.getVersion()} at ${new Date().toISOString()}
`)

  return new Promise((resolve, reject) => {
    serverProcess = spawn(executable, ['--data-dir', dir], {
      cwd: path.dirname(executable),
      env: { ...process.env, PYTHONUNBUFFERED: '1', HF_HUB_DISABLE_SYMLINKS_WARNING: '1' },
      windowsHide: true,
    })

    let port = null
    const onData = (chunk) => {
      const text = chunk.toString()
      log.write(text)
      // The server prints its chosen port before it starts listening.
      const match = text.match(/MEET_COMPANION_PORT=(\d+)/)
      if (match && port === null) {
        port = Number(match[1])
        waitForHealth(port, 60_000).then(() => resolve(port), reject)
      }
    }
    serverProcess.stdout.on('data', onData)
    serverProcess.stderr.on('data', onData)
    serverProcess.on('error', reject)
    serverProcess.on('exit', (code) => {
      log.write(`[meet-companion] server exited with code ${code}
`)
      if (port === null) reject(new Error(`The server exited early with code ${code}. See ${log.path}.`))
    })
  })
}

function stopServer() {
  if (!serverProcess || serverProcess.exitCode !== null) return
  serverProcess.kill()
  serverProcess = null
}

/**
 * Hand the server proof that this is the app on this machine, so the person
 * who installed it is not asked to sign in every launch.
 *
 * The key is a file the server wrote into its own data directory - readable
 * here, unreachable for anyone arriving over a tunnel. "Came from localhost"
 * would not do: a tunnel daemon runs locally too, so forwarded requests also
 * arrive from 127.0.0.1.
 */
async function attachDeviceKey(port) {
  try {
    // The server keeps its state in <data-dir>/data (see desktop_entry.py,
    // which points MEET_COMPANION_CONFIG there); the older layout put it at
    // the top, so accept either rather than silently not signing in.
    const candidates = [
      path.join(dataDir(), 'data', 'device.key'),
      path.join(dataDir(), 'device.key'),
    ]
    const keyPath = candidates.find((candidate) => fs.existsSync(candidate))
    if (!keyPath) return
    const key = fs.readFileSync(keyPath, 'utf8').trim()
    if (!key) return
    await session.defaultSession.cookies.set({
      url: `http://127.0.0.1:${port}`,
      name: 'mc_device',
      value: key,
      httpOnly: true,
      sameSite: 'lax',
    })
  } catch {
    // No key yet (first run races the server's own start-up), or unreadable.
    // Signing in by hand still works, so this is never fatal.
  }
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'Meet Companion',
    backgroundColor: '#f5f6fa',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  // Links to other sites open in the system browser, never inside the app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(`http://127.0.0.1:${port}`)) {
      shell.openExternal(url)
      return { action: 'deny' }
    }
    return { action: 'allow' }
  })

  attachDeviceKey(port).finally(() => mainWindow?.loadURL(`http://127.0.0.1:${port}/`))
  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(async () => {
  try {
    serverPort = await startServer()
    createWindow(serverPort)
  } catch (error) {
    dialog.showErrorBox('Meet Companion could not start', String(error.message || error))
    app.quit()
  }
})

app.on('window-all-closed', () => {
  // On macOS apps stay alive without windows; the server stays with them.
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  // Reuse the running server rather than starting a second one.
  if (BrowserWindow.getAllWindows().length === 0 && serverPort) createWindow(serverPort)
})

app.on('before-quit', stopServer)
app.on('will-quit', stopServer)
process.on('exit', stopServer)
