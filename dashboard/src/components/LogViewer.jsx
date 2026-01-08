import React, { useState } from 'react'
import apiService from '../services/api'
import './LogViewer.css'

function LogViewer({ onTestSequence }) {
  const [sequence, setSequence] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInputChange = (e) => {
    setSequence(e.target.value)
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Parse sequence from comma-separated or space-separated string
      const seqArray = sequence
        .split(/[,\s]+/)
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n))

      if (seqArray.length === 0) {
        throw new Error('Please enter at least one valid number')
      }

      const response = await apiService.scoreSequence(seqArray)
      setResult(response)

      // If it's an alert, notify parent
      if (response.alert && onTestSequence) {
        onTestSequence(seqArray)
      }
    } catch (err) {
      setError(err.message || 'Failed to score sequence')
      console.error('Error scoring sequence:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setSequence('')
    setResult(null)
    setError(null)
  }

  const severityColors = {
    HIGH: '#f44336',
    MED: '#ff9800',
    LOW: '#ffc107',
    NONE: '#4caf50',
  }

  return (
    <div className="log-viewer">
      <form onSubmit={handleSubmit} className="log-form">
        <div className="form-group">
          <label htmlFor="sequence-input">
            Enter Log Sequence (comma or space separated template IDs):
          </label>
          <textarea
            id="sequence-input"
            value={sequence}
            onChange={handleInputChange}
            placeholder="Example: 1, 2, 3, 4, 5 or 1 2 3 4 5"
            rows={3}
            className="sequence-input"
          />
        </div>
        <div className="form-actions">
          <button type="submit" disabled={loading || !sequence.trim()}>
            {loading ? '⏳ Analyzing...' : '🔍 Analyze Sequence'}
          </button>
          <button type="button" onClick={handleClear} className="btn-clear">
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="result-panel">
          <h3>Analysis Result</h3>
          <div className="result-content">
            <div className="result-item">
              <span className="result-label">Anomaly Score:</span>
              <span className="result-value">{result.score.toFixed(4)}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Severity:</span>
              <span
                className="result-value severity-badge"
                style={{ backgroundColor: severityColors[result.severity] }}
              >
                {result.severity}
              </span>
            </div>
            <div className="result-item">
              <span className="result-label">Alert:</span>
              <span className={`result-value ${result.alert ? 'alert-yes' : 'alert-no'}`}>
                {result.alert ? '⚠️ YES' : '✓ NO'}
              </span>
            </div>
            {result.predictions && result.predictions.length > 0 && (
              <div className="result-predictions">
                <strong>Top Predictions:</strong>
                <ul>
                  {result.predictions.map((pred, idx) => (
                    <li key={idx}>
                      Template ID {pred.template_id}: {(pred.probability * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default LogViewer
