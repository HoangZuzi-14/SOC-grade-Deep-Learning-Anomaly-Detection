import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const apiService = {
  // Health check
  async getHealth() {
    const response = await api.get('/')
    return response.data
  },

  // Model info
  async getModelInfo() {
    const response = await api.get('/api/v1/model/info')
    return response.data
  },

  // Score single sequence
  async scoreSequence(sequence, modelType = 'lstm') {
    const response = await api.post('/api/v1/score', {
      sequence,
      model_type: modelType,
    })
    return response.data
  },

  // Score batch
  async scoreBatch(sequences, modelType = 'lstm') {
    const response = await api.post('/api/v1/score/batch', {
      sequences,
      model_type: modelType,
    })
    return response.data
  },

  // Get event info
  async getEventInfo(eventId) {
    try {
      const response = await api.get(`/api/v1/events/${eventId}`)
      return response.data
    } catch (error) {
      return null
    }
  },

  // Alerts
  async createAlert(alert) {
    const response = await api.post('/api/v1/alerts', alert)
    return response.data
  },

  async listAlerts(limit = 100, severity = null) {
    const params = { limit }
    if (severity) params.severity = severity
    const response = await api.get('/api/v1/alerts', { params })
    return response.data
  },
}

export default apiService
