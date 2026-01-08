# SOC Anomaly Detection Dashboard

React-based web dashboard for real-time monitoring of log anomaly detection system.

## Features

- 📊 **Real-time Monitoring**: Auto-refresh with configurable intervals
- 📈 **Interactive Charts**: Score trends and threshold visualization
- 🚨 **Alert Management**: Filter and view alerts by severity
- 🔍 **Sequence Tester**: Test log sequences in real-time
- 📱 **Responsive Design**: Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### 3. Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Configuration

### API URL

By default, the dashboard connects to `http://localhost:8000`. To change this:

1. Create a `.env` file in the `dashboard/` directory:
```env
VITE_API_URL=http://your-api-url:8000
```

2. Or modify `vite.config.js` proxy settings

## Project Structure

```
dashboard/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx      # Main dashboard component
│   │   ├── AlertList.jsx      # Alert list and filtering
│   │   ├── ScoreChart.jsx     # Score visualization
│   │   ├── LogViewer.jsx      # Sequence testing interface
│   │   ├── StatsPanel.jsx     # Statistics panel
│   │   └── ModelInfo.jsx      # Model information
│   ├── services/
│   │   └── api.js             # API service layer
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## API Integration

The dashboard uses the following API endpoints:

- `GET /` - Health check
- `GET /api/v1/model/info` - Model information
- `POST /api/v1/score` - Score a sequence
- `GET /api/v1/alerts` - List alerts
- `POST /api/v1/alerts` - Create alert

## Features in Detail

### Real-time Monitoring

- Auto-refresh every 2-30 seconds (configurable)
- Manual refresh button
- Online/offline status indicator

### Alert Management

- Filter by severity (ALL, HIGH, MED, LOW)
- View alert details (sequence, score, predictions)
- Click to expand/collapse alert details

### Score Visualization

- Time-series chart of anomaly scores
- Threshold lines (P95, P99, P99.9)
- Statistics (average, max, min)
- Interactive tooltips

### Sequence Tester

- Enter log sequence as comma or space-separated IDs
- Real-time scoring
- View predictions and severity

## Development

### Adding New Components

1. Create component in `src/components/`
2. Add corresponding CSS file
3. Import and use in `Dashboard.jsx`

### Styling

- Uses CSS modules (separate `.css` files)
- Responsive design with flexbox/grid
- Color scheme: Purple gradient (#667eea to #764ba2)

### API Service

All API calls go through `src/services/api.js`. To add new endpoints:

```javascript
async newEndpoint(params) {
  const response = await api.get('/api/v1/new-endpoint', { params })
  return response.data
}
```

## Troubleshooting

### Dashboard not connecting to API

1. Check API server is running on port 8000
2. Check CORS settings in API
3. Verify `VITE_API_URL` in `.env` file

### Charts not displaying

1. Check browser console for errors
2. Verify Recharts is installed: `npm install recharts`
3. Check data format matches expected structure

### Build errors

1. Clear node_modules: `rm -rf node_modules && npm install`
2. Check Node.js version (requires 16+)
3. Verify all dependencies in `package.json`

## Production Deployment

### Build

```bash
npm run build
```

### Serve with Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/dashboard/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

### Docker (optional)

See main project `docker-compose.yml` for full stack deployment.
