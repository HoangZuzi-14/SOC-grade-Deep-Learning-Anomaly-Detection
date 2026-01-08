import React from 'react'
import './StatsPanel.css'

function StatsPanel({ alerts, scores }) {
  const severityCounts = {
    HIGH: alerts.filter((a) => a.severity === 'HIGH').length,
    MED: alerts.filter((a) => a.severity === 'MED').length,
    LOW: alerts.filter((a) => a.severity === 'LOW').length,
    NONE: alerts.filter((a) => a.severity === 'NONE').length,
  }

  const totalAlerts = alerts.length
  const avgScore = scores.length > 0
    ? scores.reduce((sum, s) => sum + s.score, 0) / scores.length
    : 0

  const recentAlerts = alerts.slice(0, 5).length

  return (
    <div className="stats-panel">
      <h3>📊 Statistics</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">🚨</div>
          <div className="stat-info">
            <div className="stat-value">{totalAlerts}</div>
            <div className="stat-label">Total Alerts</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-info">
            <div className="stat-value">{avgScore.toFixed(2)}</div>
            <div className="stat-label">Avg Score</div>
          </div>
        </div>

        <div className="stat-card severity-high">
          <div className="stat-icon">🔴</div>
          <div className="stat-info">
            <div className="stat-value">{severityCounts.HIGH}</div>
            <div className="stat-label">High Severity</div>
          </div>
        </div>

        <div className="stat-card severity-med">
          <div className="stat-icon">🟠</div>
          <div className="stat-info">
            <div className="stat-value">{severityCounts.MED}</div>
            <div className="stat-label">Medium Severity</div>
          </div>
        </div>
      </div>

      <div className="severity-breakdown">
        <h4>Severity Breakdown</h4>
        <div className="breakdown-list">
          <div className="breakdown-item">
            <span className="breakdown-label">HIGH</span>
            <div className="breakdown-bar">
              <div
                className="breakdown-fill high"
                style={{ width: `${totalAlerts > 0 ? (severityCounts.HIGH / totalAlerts) * 100 : 0}%` }}
              />
            </div>
            <span className="breakdown-value">{severityCounts.HIGH}</span>
          </div>
          <div className="breakdown-item">
            <span className="breakdown-label">MED</span>
            <div className="breakdown-bar">
              <div
                className="breakdown-fill med"
                style={{ width: `${totalAlerts > 0 ? (severityCounts.MED / totalAlerts) * 100 : 0}%` }}
              />
            </div>
            <span className="breakdown-value">{severityCounts.MED}</span>
          </div>
          <div className="breakdown-item">
            <span className="breakdown-label">LOW</span>
            <div className="breakdown-bar">
              <div
                className="breakdown-fill low"
                style={{ width: `${totalAlerts > 0 ? (severityCounts.LOW / totalAlerts) * 100 : 0}%` }}
              />
            </div>
            <span className="breakdown-value">{severityCounts.LOW}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatsPanel
