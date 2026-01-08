import React, { useState } from 'react'
import apiService from '../services/api'
import './ExplanationViewer.css'

function ExplanationViewer({ sequence, onClose }) {
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [method, setMethod] = useState('shap')
  const [error, setError] = useState(null)

  const handleExplain = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/explain/sequence', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sequence: sequence,
          method: method,
          num_samples: 100
        })
      })
      
      if (!response.ok) {
        throw new Error('Explanation failed')
      }
      
      const data = await response.json()
      setExplanation(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="explanation-viewer">
      <div className="explanation-header">
        <h3>🔍 Model Explanation</h3>
        <button onClick={onClose} className="btn-close">×</button>
      </div>

      <div className="explanation-controls">
        <label>
          Method:
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            <option value="shap">SHAP</option>
            <option value="attention">Attention</option>
          </select>
        </label>
        <button onClick={handleExplain} disabled={loading || !sequence}>
          {loading ? 'Analyzing...' : 'Explain Sequence'}
        </button>
      </div>

      {error && (
        <div className="error-message">Error: {error}</div>
      )}

      {explanation && (
        <div className="explanation-content">
          <div className="explanation-summary">
            <h4>Summary</h4>
            <div className="summary-item">
              <span>Score:</span>
              <strong>{explanation.score?.toFixed(4) || 'N/A'}</strong>
            </div>
            <div className="summary-item">
              <span>Target:</span>
              <strong>Template {explanation.target || 'N/A'}</strong>
            </div>
          </div>

          <div className="feature-importance">
            <h4>Feature Importance</h4>
            <div className="importance-list">
              {explanation.feature_importance?.slice(0, 10).map((feat, idx) => (
                <div
                  key={idx}
                  className={`importance-item ${feat.contribution > 0 ? 'positive' : 'negative'}`}
                >
                  <div className="importance-header">
                    <span>Position {feat.position}</span>
                    <span>Template {feat.template_id}</span>
                  </div>
                  <div className="importance-bar">
                    <div
                      className="importance-fill"
                      style={{
                        width: `${Math.abs(feat.contribution) * 100}%`,
                        backgroundColor: feat.contribution > 0 ? '#4caf50' : '#f44336'
                      }}
                    />
                  </div>
                  <div className="importance-value">
                    {feat.contribution > 0 ? '+' : ''}{feat.contribution.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {explanation.top_predictions && (
            <div className="top-predictions">
              <h4>Top Predictions</h4>
              <ul>
                {explanation.top_predictions.map((pred, idx) => (
                  <li key={idx}>
                    Template {pred.template_id}: {(pred.probability * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ExplanationViewer
