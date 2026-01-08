# REST API Documentation

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API server
```bash
cd api
python main.py
```

Or with uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access API
- API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/

## Environment Variables

```bash
export MODEL_PATH="model/model.pth"
export SCORES_PATH="data/sequences/lstm_scores.pkl"
export MAPPING_PATH="data/sequences/event_mapping.json"
export DEVICE="cpu"  # or "cuda"
```

## API Endpoints

### Health Check
```
GET /
```

### Score Single Sequence
```
POST /api/v1/score
Body: {
    "sequence": [1, 2, 3, 4, 5],
    "model_type": "lstm"
}
```

### Score Batch
```
POST /api/v1/score/batch
Body: {
    "sequences": [[1,2,3], [4,5,6]],
    "model_type": "lstm"
}
```

### Model Info
```
GET /api/v1/model/info
```

### Get Event Info
```
GET /api/v1/events/{event_id}
```

### Create Alert
```
POST /api/v1/alerts
Body: {
    "sequence": [1, 2, 3],
    "score": 5.2,
    "severity": "HIGH"
}
```

### List Alerts
```
GET /api/v1/alerts?limit=100&severity=HIGH
```

## Testing with curl

```bash
# Health check
curl http://localhost:8000/

# Score sequence
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"sequence": [1, 2, 3, 4, 5]}'

# Model info
curl http://localhost:8000/api/v1/model/info
```
