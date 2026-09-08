"""
Evaluation Judge Logic.
This file implements the `LLMJudge` classes responsible for scoring and grading
generated answers against provided rubrics or expected outcomes.
"""

import json
import logging
import re
from config.settings import settings


logger = logging.getLogger(__name__)


class LLMJudge:
    def __init__(self, model: str = None):
        self.source = getattr(settings, "JUDGE_SOURCE", "openai").lower()

        if self.source == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL
            )
            self.model = model or getattr(settings, "OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        elif self.source == "huggingface":
            from huggingface_hub import AsyncInferenceClient

            self.client = AsyncInferenceClient(token=settings.HUGGINGFACE_API_KEY)
            self.model = model or getattr(
                settings,
                "HUGGINGFACE_JUDGE_MODEL",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            )
        elif self.source == "ollama":
            import ollama

            # Initialize async client (assumes local ollama on default port, or use settings.OLLAMA_BASE_URL)
            base_url = getattr(settings, "OLLAMA_BASE_URL", None)
            if base_url:
                self.client = ollama.AsyncClient(host=base_url)
            else:
                self.client = ollama.AsyncClient()
            self.model = model or getattr(settings, "OLLAMA_JUDGE_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported Judge Source: {self.source}")

    async def _generate_combined_scores(self, prompt: str) -> dict:
        try:
            if self.source == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.OPENAI_TEMPERATURE,
                )
                content = response.choices[0].message.content

            elif self.source == "huggingface":
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                    max_tokens=150,
                )
                content = response.choices[0].message.content

            elif self.source == "ollama":
                response = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": settings.OLLAMA_TEMPERATURE},
                )
                content = response["message"]["content"]

            default_scores = {"task_success": 0.0, "groundedness": 0.0, "retrieval_hit": 0.0}
            
            try:
                # first try pure json parsing, finding the { block
                match = re.search(r'\{.*\}', content.replace('\n', ' '))
                if match:
                    parsed = json.loads(match.group(0))
                    for k in default_scores:
                        if k in parsed:
                            default_scores[k] = float(parsed[k])
                    return default_scores
            except:
                pass
                
            # fallback regex
            for k in default_scores:
                match = re.search(f'"{k}"\\s*:\\s*([0-9.]+)', content)
                if match:
                    default_scores[k] = float(match.group(1))
            return default_scores
        except Exception as e:
            logger.error(f"Error in judge generation: {e}")
            return {"task_success": 0.0, "groundedness": 0.0, "retrieval_hit": 0.0}

    async def evaluate_all(
        self, query: str, context: list, generated_answer: str, expected_answer: str
    ) -> dict:
        from rag.prompts.prompts import build_comprehensive_eval_prompt
        context_str = "\n\n".join([c.get("chunk", str(c)) if isinstance(c, dict) else str(c) for c in context])
        prompt = build_comprehensive_eval_prompt(query, expected_answer, context_str, generated_answer)
        return await self._generate_combined_scores(prompt)
