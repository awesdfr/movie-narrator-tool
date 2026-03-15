class WebSocketService {
  constructor() {
    this.ws = null
    this.projectId = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 3000
    this.listeners = new Map()
    this.isConnecting = false
  }

  connect(projectId) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      if (this.projectId === projectId) {
        this.requestProgress()
        return Promise.resolve()
      }
      this.disconnect()
    }

    if (this.isConnecting) {
      return Promise.resolve()
    }

    this.isConnecting = true
    this.projectId = projectId

    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws/project/${projectId}`

      this.ws = new WebSocket(wsUrl)

      this.ws.onopen = () => {
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.requestProgress()
        resolve()
      }

      this.ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data)
          this.handleMessage(data)
        } catch (error) {
          console.error('Failed to parse websocket message', error)
        }
      }

      this.ws.onclose = () => {
        this.isConnecting = false
        this.tryReconnect()
      }

      this.ws.onerror = error => {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.projectId = null
    this.reconnectAttempts = 0
  }

  tryReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      return
    }

    if (!this.projectId) {
      return
    }

    this.reconnectAttempts += 1
    window.setTimeout(() => {
      this.connect(this.projectId)
    }, this.reconnectDelay)
  }

  handleMessage(data) {
    const { type } = data

    if (type === 'ping') {
      this.send({ type: 'pong' })
      return
    }

    if (this.listeners.has(type)) {
      this.listeners.get(type).forEach(callback => callback(data))
    }

    if (this.listeners.has('*')) {
      this.listeners.get('*').forEach(callback => callback(data))
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  on(type, callback) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, [])
    }
    this.listeners.get(type).push(callback)

    return () => {
      const callbacks = this.listeners.get(type) || []
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
      if (callbacks.length === 0) {
        this.listeners.delete(type)
      }
    }
  }

  off(type, callback) {
    if (!this.listeners.has(type)) {
      return
    }
    const callbacks = this.listeners.get(type)
    const index = callbacks.indexOf(callback)
    if (index > -1) {
      callbacks.splice(index, 1)
    }
    if (callbacks.length === 0) {
      this.listeners.delete(type)
    }
  }

  requestProgress() {
    this.send({ type: 'get_progress' })
  }
}

export const wsService = new WebSocketService()
export default wsService
