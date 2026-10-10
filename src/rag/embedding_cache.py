"""
Embedding Cache Implementation.
Caches exact text chunks and their dense embeddings to avoid redundant LLM/API calls.
"""

import json
import logging
import os
import uuid
import hashlib
from datetime import datetime, timezone

from config.settings import settings

logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_uuid_from_text(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_text(text)))

class EmbeddingCache:
    def __init__(self):
        self.store_type = getattr(settings, "VECTOR_STORE_TYPE", "json").lower()
        self.enabled = getattr(settings, "ENABLE_EMBEDDING_CACHE", False)

        if not self.enabled:
            return

        if self.store_type == "json":
            self.filepath = getattr(
                settings, "EMBEDDING_CACHE_JSON_PATH", "data/vector_store_ecache.json"
            )
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if not os.path.exists(self.filepath):
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        elif self.store_type == "chromadb":
            import chromadb

            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(name="ecache")
        elif self.store_type == "qdrant":
            from qdrant_client import QdrantClient

            self.client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
            )
            self.collection_name = "ecache"
            self.collection_exist = self.client.collection_exists(self.collection_name)

    async def get(self, text: str) -> list[float]:
        if not self.enabled:
            return None

        try:
            if self.store_type == "json":
                return self._get_json(text)
            elif self.store_type == "chromadb":
                return self._get_chroma(text)
            elif self.store_type == "qdrant":
                return self._get_qdrant(text)
        except Exception as e:
            logger.error(f"Error accessing embedding cache: {e}")

        return None

    def _get_json(self, text: str) -> list[float]:
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return None

        key = hash_text(text)
        if key in data:
            logger.info("Embedding cache hit (JSON).")
            return data[key].get("embedding")
            
        return None

    def _get_chroma(self, text: str) -> list[float]:
        point_id = get_uuid_from_text(text)
        result = self.collection.get(ids=[point_id], include=["embeddings"])
        
        if result and result.get("embeddings") and len(result["embeddings"]) > 0:
            logger.info("Embedding cache hit (ChromaDB).")
            return result["embeddings"][0]
            
        return None

    def _get_qdrant(self, text: str) -> list[float]:
        if not self.collection_exist:
            return None

        try:
            point_id = get_uuid_from_text(text)
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True
            )
            if records:
                logger.info("Embedding cache hit (Qdrant).")
                vector = records[0].vector
                if isinstance(vector, dict) and "dense" in vector:
                    return vector["dense"]
                elif isinstance(vector, list):
                    return vector
        except Exception as e:
            logger.warning(f"Qdrant cache retrieve failed: {e}")
        return None

    async def set(self, text: str, embedding: list[float]):
        if not self.enabled:
            return

        try:
            if self.store_type == "json":
                self._set_json(text, embedding)
            elif self.store_type == "chromadb":
                self._set_chroma(text, embedding)
            elif self.store_type == "qdrant":
                self._set_qdrant(text, embedding)
        except Exception as e:
            logger.error(f"Error updating embedding cache: {e}")

    def _set_json(self, text: str, embedding: list[float]):
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        now = datetime.now(timezone.utc).isoformat()
        ttl = getattr(settings, "EMBEDDING_CACHE_TTL_HOURS", 24.0)

        key = hash_text(text)
        data[key] = {
            "text": text,
            "embedding": embedding,
            "created_at": now,
            "ttl_hours": ttl,
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _set_chroma(self, text: str, embedding: list[float]):
        now = datetime.now(timezone.utc).isoformat()
        ttl = getattr(settings, "EMBEDDING_CACHE_TTL_HOURS", 24.0)
        point_id = get_uuid_from_text(text)
        
        self.collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "text": text,
                "created_at": now,
                "ttl_hours": ttl,
            }],
            ids=[point_id],
        )

    def _set_qdrant(self, text: str, embedding: list[float]):
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not self.collection_exist:
            vector_size = len(embedding)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                },
            )
            self.collection_exist = True

        now = datetime.now(timezone.utc).isoformat()
        ttl = getattr(settings, "EMBEDDING_CACHE_TTL_HOURS", 24.0)
        point_id = get_uuid_from_text(text)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"dense": embedding},
                    payload={
                        "text": text,
                        "created_at": now,
                        "ttl_hours": ttl,
                    },
                )
            ],
        )
