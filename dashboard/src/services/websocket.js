/**
 * WebSocket service for real-time streaming
 */
class WebSocketService {
  constructor() {
    this.ws = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 3000
    this.listeners = new Map()
    this.isConnected = false
  }

  connect(url = null) {
    const wsUrl = url || import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/streaming/ws'
    
    try {
      this.ws = new WebSocket(wsUrl)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.isConnected = true
        this.reconnectAttempts = 0
        this.emit('connection', { status: 'connected' })
        
        // Send ping
        this.send({ type: 'ping' })
      }
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.handleMessage(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.emit('error', error)
      }
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.isConnected = false
        this.emit('connection', { status: 'disconnected' })
        this.attemptReconnect()
      }
    } catch (error) {
      console.error('Error connecting WebSocket:', error)
      this.attemptReconnect()
    }
  }

  handleMessage(data) {
    const type = data.type
    
    // Emit to all listeners
    this.emit(type, data)
    
    // Handle specific message types
    switch (type) {
      case 'score':
        this.emit('score', data.data)
        break
      case 'alert':
        this.emit('alert', data.data)
        break
      case 'stats':
        this.emit('stats', data.data)
        break
      case 'pong':
        // Keepalive
        break
      default:
        this.emit('message', data)
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event).push(callback)
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in ${event} listener:`, error)
        }
      })
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`)
      
      setTimeout(() => {
        this.connect()
      }, this.reconnectDelay)
    } else {
      console.error('Max reconnection attempts reached')
      this.emit('error', { message: 'Max reconnection attempts reached' })
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isConnected = false
    this.listeners.clear()
  }

  getConnectionStatus() {
    return this.isConnected
  }
}

// Export singleton instance
const wsService = new WebSocketService()
export default wsService
