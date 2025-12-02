import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """
    Wrapper around a persistent ChromaDB index for semantic retrieval.
    Uses the same embedding model that was used in Colab
    ('all-MiniLM-L6-v2') so that queries are compatible.
    """

    def __init__(
        self,
        index_path: str = os.path.join("rag", "index"),
        collection_name: str = "traffic_knowledge",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        :param index_path: Path to the ChromaDB persistent directory.
        :param collection_name: Name of the collection inside Chroma.
        :param embedding_model_name: SentenceTransformer model name.
        """
        self.index_path = index_path
        self.collection_name = collection_name

        # Load embedding model (same as in Colab)
        self.embedder = SentenceTransformer(embedding_model_name)

        # Create Chroma persistent client pointing to the existing index
        self.client = chromadb.PersistentClient(path=self.index_path)

        # Get the collection (must already exist from Colab)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def _embed_query(self, query: str) -> List[float]:
        """Convert a text query into an embedding vector (as a Python list)."""
        emb = self.embedder.encode([query])[0]
        return emb.tolist()

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a given query.

        :param query: Natural language query text.
        :param k: Number of results to return.
        :return: List of dicts: { 'text', 'metadata', 'score' }
        """
        if not query or not query.strip():
            return []

        query_emb = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)

        formatted = []
        for text, meta, dist in zip(docs, metas, dists):
            formatted.append(
                {
                    "text": text,
                    "metadata": meta,
                    "score": dist,  # smaller distance = closer match
                }
            )

        return formatted