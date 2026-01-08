import React, { useState, useEffect } from 'react'
import AlertList from './AlertList'
import ScoreChart from './ScoreChart'
import LogViewer from './LogViewer'
import StatsPanel from './StatsPanel'
import ModelInfo from './ModelInfo'
import apiService from '../services/api'
import wsService from '../services/websocket'
import './Dashboard.css'

function Dashboard() {
  const [health, setHealth] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [scores, setScores] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5000) // 5 seconds
  const [wsConnected, setWsConnected] = useState(false)
  const [streamingActive, setStreamingActive] = useState(false)

  // Load initial data
  useEffect(() => {
    loadData()
    
    // Setup WebSocket connection
    wsService.connect()
    
    // WebSocket event handlers
    wsService.on('connection', (data) => {
      setWsConnected(data.status === 'connected')
    })
    
    wsService.on('alert', (alertData) => {
      // Add new alert to list
      const newAlert = {
        id: alertData.id || `alert_${Date.now()}`,
        sequence: alertData.sequence,
        score: alertData.score,
        severity: alertData.severity,
        timestamp: alertData.created_at || new Date().toISOString(),
      }
      setAlerts((prev) => [newAlert, ...prev].slice(0, 50))
    })
    
    wsService.on('score', (scoreData) => {
      // Update scores for chart
      const newScore = {
        time: new Date().toISOString(),
        score: scoreData.score,
        severity: scoreData.severity,
      }
      setScores((prev) => [...prev.slice(-24), newScore])
    })
    
    wsService.on('stats', (statsData) => {
      // Update statistics
      console.log('Stats update:', statsData)
    })
    
    // Cleanup
    return () => {
      wsService.disconnect()
    }
  }, [])

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      loadData()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  const loadData = async () => {
    try {
      setIsLoading(true)
      setError(null)

      // Load health and model info
      const [healthData, modelData, alertsData] = await Promise.all([
        apiService.getHealth(),
        apiService.getModelInfo().catch(() => null),
        apiService.listAlerts(50).catch(() => ({ alerts: [], count: 0 })),
      ])

      setHealth(healthData)
      setModelInfo(modelData)
      setAlerts(alertsData.alerts || [])

      // Generate sample scores for visualization (in real app, get from API)
      if (scores.length === 0) {
        generateSampleScores()
      }
    } catch (err) {
      setError(err.message)
      console.error('Error loading data:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const generateSampleScores = () => {
    // Generate sample time-series data for visualization
    const now = new Date()
    const sampleScores = []
    for (let i = 24; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000)
      sampleScores.push({
        time: time.toISOString(),
        score: Math.random() * 10,
        severity: ['NONE', 'LOW', 'MED', 'HIGH'][Math.floor(Math.random() * 4)],
      })
    }
    setScores(sampleScores)
  }

  const handleTestSequence = async (sequence) => {
    try {
      const result = await apiService.scoreSequence(sequence)
      // Add to alerts if it's an alert
      if (result.alert) {
        const newAlert = {
          id: `alert_${Date.now()}`,
          sequence: result.sequence,
          score: result.score,
          severity: result.severity,
          timestamp: new Date().toISOString(),
        }
        setAlerts((prev) => [newAlert, ...prev].slice(0, 50))
      }
      return result
    } catch (err) {
      setError(err.message)
      throw err
    }
  }

  if (isLoading && !health) {
    return (
      <div className="dashboard-loading">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🔒 SOC Anomaly Detection Dashboard</h1>
        <div className="header-controls">
          <label>
            Auto-refresh:
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
          </label>
          {autoRefresh && (
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
            >
              <option value={2000}>2s</option>
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
            </select>
          )}
          <button onClick={loadData} className="btn-refresh">
            🔄 Refresh
          </button>
          <div className={`status-indicator ${health?.status === 'ok' ? 'online' : 'offline'}`}>
            {health?.status === 'ok' ? '🟢 Online' : '🔴 Offline'}
          </div>
          <div className={`status-indicator ${wsConnected ? 'online' : 'offline'}`}>
            {wsConnected ? '📡 WS Connected' : '📡 WS Disconnected'}
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          ⚠️ Error: {error}
        </div>
      )}

      <div className="dashboard-content">
        <div className="dashboard-sidebar">
          <ModelInfo modelInfo={modelInfo} health={health} />
          <StatsPanel alerts={alerts} scores={scores} />
        </div>

        <div className="dashboard-main">
          <div className="dashboard-section">
            <h2>📊 Anomaly Score Trends</h2>
            <ScoreChart scores={scores} />
          </div>

          <div className="dashboard-section">
            <h2>🚨 Active Alerts</h2>
            <AlertList alerts={alerts} onTestSequence={handleTestSequence} />
          </div>

          <div className="dashboard-section">
            <h2>📝 Log Sequence Tester</h2>
            <LogViewer onTestSequence={handleTestSequence} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
