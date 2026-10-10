"""
Semantic Cache Implementation.
Caches queries and their answers in a vector store to speed up response times for similar queries.
"""

import json
import logging
import math
import os
import uuid

from config.settings import settings
from rag.embeddings import ModelSelector

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self):
        self.store_type = getattr(settings, "VECTOR_STORE_TYPE", "json").lower()
        self.threshold = getattr(settings, "SEMANTIC_CACHE_THRESHOLD", 0.9)
        self.enabled = getattr(settings, "ENABLE_SEMANTIC_CACHE", False)

        if not self.enabled:
            return

        if self.store_type == "json":
            self.filepath = getattr(
                settings, "SEMANTIC_CACHE_JSON_PATH", "data/vector_store_cache.json"
            )
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if not os.path.exists(self.filepath):
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump([], f)
        elif self.store_type == "chromadb":
            import chromadb

            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(name="cache")
        elif self.store_type == "qdrant":
            from qdrant_client import QdrantClient

            self.client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
            )
            self.collection_name = "cache"
            self.collection_exist = self.client.collection_exists(self.collection_name)

    async def get(self, query: str) -> str:
        if not self.enabled:
            return None

        try:
            query_embedding = await ModelSelector.get_single_embedding(query)

            if self.store_type == "json":
                return self._get_json(query_embedding)
            elif self.store_type == "chromadb":
                return self._get_chroma(query_embedding)
            elif self.store_type == "qdrant":
                return self._get_qdrant(query_embedding)
        except Exception as e:
            logger.error(f"Error accessing semantic cache: {e}")

        return None

    def _get_json(self, query_embedding: list[float]) -> str:
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return None

        best_score = -1
        best_answer = None

        for item in data:
            chunk_embed = item.get("embedding")
            if chunk_embed and len(chunk_embed) == len(query_embedding):
                dot_product = sum(a * b for a, b in zip(chunk_embed, query_embedding))
                norm_a = math.sqrt(sum(a * a for a in chunk_embed))
                norm_b = math.sqrt(sum(b * b for b in query_embedding))
                similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

                if similarity > best_score:
                    best_score = similarity
                    best_answer = item.get("answer")

        if best_score >= self.threshold:
            logger.info(f"Semantic cache hit (score: {best_score:.4f})")
            return best_answer
        return None

    def _get_chroma(self, query_embedding: list[float]) -> str:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=1)
        if result["distances"] and result["distances"][0]:
            distance = result["distances"][0][0]
            similarity = 1.0 - distance
            if similarity >= self.threshold:
                logger.info(f"Semantic cache hit (score: {similarity:.4f})")
                return result["metadatas"][0][0].get("answer")
        return None

    def _get_qdrant(self, query_embedding: list[float]) -> str:
        if not self.collection_exist:
            return None

        from qdrant_client import models

        try:
            try:
                temp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    using="dense",
                    limit=1,
                )
            except Exception:
                temp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=1,
                )

            if temp.points:
                point = temp.points[0]
                if point.score >= self.threshold:
                    logger.info(f"Semantic cache hit (score: {point.score:.4f})")
                    return point.payload.get("answer")
        except Exception as e:
            logger.warning(f"Qdrant cache search failed: {e}")
        return None

    async def set(self, query: str, answer: str):
        if not self.enabled:
            return

        try:
            query_embedding = await ModelSelector.get_single_embedding(query)

            if self.store_type == "json":
                self._set_json(query, query_embedding, answer)
            elif self.store_type == "chromadb":
                self._set_chroma(query, query_embedding, answer)
            elif self.store_type == "qdrant":
                self._set_qdrant(query, query_embedding, answer)
        except Exception as e:
            logger.error(f"Error updating semantic cache: {e}")

    def _set_json(self, query: str, query_embedding: list[float], answer: str):
        data = []
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []

        data.append({"query": query, "embedding": query_embedding, "answer": answer})

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _set_chroma(self, query: str, query_embedding: list[float], answer: str):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query))
        self.collection.upsert(
            documents=[query],
            embeddings=[query_embedding],
            metadatas=[{"answer": answer, "query": query}],
            ids=[point_id],
        )

    def _set_qdrant(self, query: str, query_embedding: list[float], answer: str):
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not self.collection_exist:
            vector_size = len(query_embedding)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                },
            )
            self.collection_exist = True

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"dense": query_embedding},
                    payload={"query": query, "answer": answer},
                )
            ],
        )
