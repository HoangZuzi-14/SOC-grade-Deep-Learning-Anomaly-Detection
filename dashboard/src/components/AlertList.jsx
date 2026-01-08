import React, { useState } from 'react'
import { format } from 'date-fns'
import ExplanationViewer from './ExplanationViewer'
import './AlertList.css'

function AlertList({ alerts, onTestSequence }) {
  const [filter, setFilter] = useState('ALL')
  const [sortBy, setSortBy] = useState('priority') // 'priority', 'score', 'time'
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [explainingAlert, setExplainingAlert] = useState(null)

  const severityColors = {
    HIGH: '#f44336',
    MED: '#ff9800',
    LOW: '#ffc107',
    NONE: '#4caf50',
  }

  const filteredAlerts = alerts.filter((alert) => {
    if (filter === 'ALL') return true
    return alert.severity === filter
  })

  const getSeverityCount = (severity) => {
    return alerts.filter((a) => a.severity === severity).length
  }

  return (
    <div className="alert-list">
      <div className="alert-filters">
        <button
          className={`filter-btn ${filter === 'ALL' ? 'active' : ''}`}
          onClick={() => setFilter('ALL')}
        >
          All ({alerts.length})
        </button>
        <button
          className={`filter-btn ${filter === 'HIGH' ? 'active' : ''}`}
          onClick={() => setFilter('HIGH')}
        >
          High ({getSeverityCount('HIGH')})
        </button>
        <button
          className={`filter-btn ${filter === 'MED' ? 'active' : ''}`}
          onClick={() => setFilter('MED')}
        >
          Medium ({getSeverityCount('MED')})
        </button>
        <button
          className={`filter-btn ${filter === 'LOW' ? 'active' : ''}`}
          onClick={() => setFilter('LOW')}
        >
          Low ({getSeverityCount('LOW')})
        </button>
      </div>

      <div className="alert-items">
        {filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            <p>No alerts found</p>
            <p className="subtext">Alerts will appear here when anomalies are detected</p>
          </div>
        ) : (
          sortedAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`alert-item ${alert.severity}`}
              onClick={() => setSelectedAlert(selectedAlert === alert.id ? null : alert.id)}
            >
              <div className="alert-header">
                <div className="alert-severity">
                  <span
                    className="severity-badge"
                    style={{ backgroundColor: severityColors[alert.severity] }}
                  >
                    {alert.severity}
                  </span>
                </div>
                <div className="alert-score">
                  Score: <strong>{alert.score?.toFixed(4) || 'N/A'}</strong>
                </div>
                <div className="alert-time">
                  {alert.timestamp
                    ? format(new Date(alert.timestamp), 'MMM dd, HH:mm:ss')
                    : 'N/A'}
                </div>
              </div>

              {selectedAlert === alert.id && (
                <div className="alert-details">
                  <div className="alert-sequence">
                    <strong>Sequence:</strong>
                    <div className="sequence-display">
                      {alert.sequence?.map((id, idx) => (
                        <span key={idx} className="sequence-item">
                          {id}
                        </span>
                      ))}
                    </div>
                  </div>
                  {alert.predictions && (
                    <div className="alert-predictions">
                      <strong>Top Predictions:</strong>
                      <ul>
                        {alert.predictions.map((pred, idx) => (
                          <li key={idx}>
                            ID {pred.template_id}: {(pred.probability * 100).toFixed(2)}%
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="alert-actions">
                    <button
                      className="btn-explain"
                      onClick={() => setExplainingAlert(alert.sequence)}
                    >
                      🔍 Explain This Sequence
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {explainingAlert && (
        <div className="explanation-modal">
          <div className="explanation-modal-content">
            <ExplanationViewer
              sequence={explainingAlert}
              onClose={() => setExplainingAlert(null)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default AlertList
