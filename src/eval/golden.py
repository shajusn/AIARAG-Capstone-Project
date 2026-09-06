"""
Golden Set Evaluation Logic.
"""

import json
import time
import asyncio
import statistics
import logging
from typing import List, Dict, Any

from pipeline.inference import InferencePipeline
from eval.judge import LLMJudge

logger = logging.getLogger(__name__)


def load_golden_set(filepath: str) -> List[Dict]:
    dataset = []
    # Using a simple JSON parser since the format is a pseudo JSON-list
    # In practice, we'll read it as a single string and parse the list.
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
            # If the file does not have wrapping brackets, wrap it
            if not content.startswith("["):
                content = f"[{content}]"
            dataset = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to load golden set: {e}")
    return dataset


async def run_evaluation(
    golden_dataset: List[Dict], pipeline: InferencePipeline, judge: LLMJudge
):
    results = []

    print(f"Starting evaluation on {len(golden_dataset)} items...")

    for item in golden_dataset:
        query = item["question"]
        expected_answer = item["ideal_answer"]
        item_level = item.get("level", "unknown")

        print(f"Evaluating [{item['id']}] ({item_level}) - {query[:50]}...")

        # 1. Measure Retrieval Latency & Hit Rate
        t0 = time.time()
        try:
            context = await pipeline.retriever.retrieve(query, top_k=pipeline.top_k)
        except Exception as e:
            logger.error(f"Retrieval failed for {query}: {e}")
            context = []
        t1 = time.time()
        retrieval_latency = t1 - t0

        # 2. Measure Generation Latency
        try:
            answer, usage_info = await pipeline.generator.generate_answer(
                query, context, return_usage=True
            )
        except Exception as e:
            logger.error(f"Generation failed for {query}: {e}")
            answer = ""
            usage_info = {"prompt_tokens": 0, "completion_tokens": 0}
        t2 = time.time()
        generation_latency = t2 - t1

        total_latency = t2 - t0

        # Approximate tokens for embedding (query only since context is pre-embedded)
        query_chars = len(query)
        embedding_tokens = query_chars / 4.0

        from config.settings import settings

        embedding_cost = embedding_tokens * (
            settings.EMBEDDING_COST_PER_1M_TOKENS / 1_000_000.0
        )
        llm_cost = (
            usage_info["prompt_tokens"]
            * (settings.LLM_INPUT_COST_PER_1M_TOKENS / 1_000_000.0)
        ) + (
            usage_info["completion_tokens"]
            * (settings.LLM_OUTPUT_COST_PER_1M_TOKENS / 1_000_000.0)
        )
        cost_per_query = embedding_cost + llm_cost

        # Evaluate with LLMJudge
        scores = await judge.evaluate_all(query, context, answer, expected_answer)
        task_success = scores["task_success"]
        groundedness = scores["groundedness"]
        retrieval_hit = scores["retrieval_hit"]

        results.append(
            {
                "id": item["id"],
                "question": query,
                "ideal_answer": expected_answer,
                "generated_answer": answer,
                "level": item_level,
                "latency": total_latency,
                "cost": cost_per_query,
                "task_success": task_success,
                "groundedness": groundedness,
                "retrieval_hit": retrieval_hit,
            }
        )

    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {}

    latencies = [r["latency"] for r in results]
    latencies.sort()

    def percentile(data, p):
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1
        if f == c:
            return data[int(k)]
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return d0 + d1

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)

    avg = lambda key: sum(r[key] for r in results) / len(results)

    return {
        "Total Count": len(results),
        "Task Success Rate": f"{avg('task_success'):.2%}",
        "Groundedness": f"{avg('groundedness'):.2%}",
        "Retrieval Hit Rate": f"{avg('retrieval_hit'):.2%}",
        "Cost / Query": f"${avg('cost'):.5f}",
        "Latency (p50)": f"{p50:.2f}s",
        "Latency (p95)": f"{p95:.2f}s",
    }
