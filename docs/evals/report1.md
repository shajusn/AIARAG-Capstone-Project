# RAG Evaluation Report

## Final Metrics vs Thresholds

| Metric | Acceptable Threshold | Good Threshold | Value | Threshold Evaluation |
|---|---|---|---|---|
| Total Count | N/A | N/A | 5 | N/A |
| Task Success Rate | 0.6 | 0.8 | 100.00% | Good |
| Groundedness | 0.8 | 0.95 | 100.00% | Good |
| Retrieval Hit Rate | 0.7 | 0.9 | 100.00% | Good |
| Cost / Query | 0.01 | 0.005 | $0.00002 | Good |
| Latency (p50) | N/A | N/A | 3.19s | N/A |
| Latency (p95) | 3.0 | 1.5 | 20.33s | Poor |

## Metrics by Path Level

| Path Level | Count | Task Success Rate | Groundedness | Retrieval Hit Rate |
|---|---|---|---|---|
| happy_path | 5 | 100.00% | 100.00% | 100.00% |

## Threshold Definitions
```json
{
  "task_success_rate": {
    "acceptable": 0.6,
    "good": 0.8
  },
  "groundedness": {
    "acceptable": 0.8,
    "good": 0.95
  },
  "retrieval_rate": {
    "acceptable": 0.7,
    "good": 0.9
  },
  "cost_per_query": {
    "acceptable": 0.01,
    "good": 0.005
  },
  "latency_p95": {
    "acceptable": 3.0,
    "good": 1.5
  }
}
```

## Cost Evaluation

| Total Questions | Topk Value | Total Input Tokens | System Prompt Tokens | Retrieved Chunks Tokens | Question Embeding Tokens | Answer Generated Tokens | Cost | Avg Latency |
|---|---|---|---|---|---|---|---|---|
| 5 | 5 | 7693.0 | 1049.0 | 6578.0 | 66.0 | 121.0 | $0.000122 | 6.94s |
