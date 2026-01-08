import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'
import { format } from 'date-fns'
import './ScoreChart.css'

function ScoreChart({ scores }) {
  // Transform data for chart
  const chartData = scores.map((item) => ({
    time: format(new Date(item.time), 'HH:mm'),
    fullTime: item.time,
    score: item.score,
    severity: item.severity,
  }))

  // Calculate statistics
  const avgScore = scores.length > 0
    ? scores.reduce((sum, s) => sum + s.score, 0) / scores.length
    : 0
  const maxScore = scores.length > 0 ? Math.max(...scores.map((s) => s.score)) : 0
  const minScore = scores.length > 0 ? Math.min(...scores.map((s) => s.score)) : 0

  // Severity thresholds (example - should come from API)
  const thresholds = {
    p95: 7.0,
    p99: 8.5,
    p999: 10.0,
  }

  const getSeverityColor = (severity) => {
    const colors = {
      HIGH: '#f44336',
      MED: '#ff9800',
      LOW: '#ffc107',
      NONE: '#4caf50',
    }
    return colors[severity] || '#999'
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="chart-tooltip">
          <p className="tooltip-time">{format(new Date(data.fullTime), 'MMM dd, yyyy HH:mm')}</p>
          <p className="tooltip-score">
            Score: <strong>{data.score.toFixed(4)}</strong>
          </p>
          <p className="tooltip-severity">
            Severity: <span style={{ color: getSeverityColor(data.severity) }}>
              {data.severity}
            </span>
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="score-chart">
      <div className="chart-stats">
        <div className="stat-item">
          <span className="stat-label">Average:</span>
          <span className="stat-value">{avgScore.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Max:</span>
          <span className="stat-value">{maxScore.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Min:</span>
          <span className="stat-value">{minScore.toFixed(4)}</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#667eea" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#667eea" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="time"
            stroke="#666"
            tick={{ fontSize: 12 }}
            interval="preserveStartEnd"
          />
          <YAxis stroke="#666" tick={{ fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#667eea"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorScore)"
            name="Anomaly Score"
          />
          {/* Threshold lines */}
          <Line
            type="monotone"
            dataKey={() => thresholds.p95}
            stroke="#ffc107"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="P95 Threshold"
          />
          <Line
            type="monotone"
            dataKey={() => thresholds.p99}
            stroke="#ff9800"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="P99 Threshold"
          />
          <Line
            type="monotone"
            dataKey={() => thresholds.p999}
            stroke="#f44336"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="P99.9 Threshold"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default ScoreChart
