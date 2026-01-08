import React from 'react'
import './ModelInfo.css'

function ModelInfo({ modelInfo, health }) {
  if (!modelInfo && !health) {
    return (
      <div className="model-info">
        <h3>🤖 Model Information</h3>
        <div className="info-item">
          <span className="info-label">Status:</span>
          <span className="info-value">Not Loaded</span>
        </div>
      </div>
    )
  }

  const isModelLoaded = health?.model_loaded || modelInfo?.model_loaded
  const config = modelInfo?.config || health?.config || {}  // Updated to match API response
  const thresholds = modelInfo?.thresholds || health?.thresholds

  return (
    <div className="model-info">
      <h3>🤖 Model Information</h3>
      <div className="info-content">
        <div className="info-item">
          <span className="info-label">Status:</span>
          <span className={`info-value ${isModelLoaded ? 'status-online' : 'status-offline'}`}>
            {isModelLoaded ? '✅ Loaded' : '❌ Not Loaded'}
          </span>
        </div>

        {isModelLoaded && (
          <>
            <div className="info-section">
              <h4>Configuration</h4>
              <div className="info-item">
                <span className="info-label">Window Size:</span>
                <span className="info-value">{config.window_size || 'N/A'}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Embedding Dim:</span>
                <span className="info-value">{config.embedding_dim || 'N/A'}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Hidden Size:</span>
                <span className="info-value">{config.hidden_size || 'N/A'}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Num Labels:</span>
                <span className="info-value">{config.num_labels || 'N/A'}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Device:</span>
                <span className="info-value">{config.device || 'N/A'}</span>
              </div>
            </div>

            {thresholds && (
              <div className="info-section">
                <h4>Thresholds</h4>
                <div className="info-item">
                  <span className="info-label">P95:</span>
                  <span className="info-value">{thresholds.p95?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">P99:</span>
                  <span className="info-value">{thresholds.p99?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">P99.9:</span>
                  <span className="info-value">{thresholds.p999?.toFixed(4) || 'N/A'}</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default ModelInfo
