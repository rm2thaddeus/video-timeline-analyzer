"""
📌 Purpose – Qdrant integration for ingesting scene-level metadata and embeddings from the canonical DataFrame, and querying for semantic search.
🔄 Latest Changes – Initial creation. Implements ingestion and query for scene-level data, following project schema and modularity rules.
⚙️ Key Logic – Connects to Qdrant, creates collection, ingests DataFrame (embedding as vector, other fields as payload), and supports vector search.
📂 Expected File Path – src/db/qdrant_client.py
🧠 Reasoning – Ensures modular, reproducible, and maintainable database integration for scientific video analysis pipelines.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

class QdrantClientWrapper:
    """
    Wrapper for Qdrant vector database integration.

    Args:
        host (str): Qdrant host (default: 'localhost').
        port (int): Qdrant port (default: 6333).
        collection_name (str): Name of the Qdrant collection.
        vector_size (int): Size of the embedding vector.
        distance (str): Distance metric ('Cosine', 'Euclid', 'Dot').
    """
    def __init__(self, collection_name: str, vector_size: int, host: str = 'localhost', port: int = 6333, distance: str = 'Cosine'):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = vector_size
        self.distance = distance
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance[self.distance])
            )

    def ingest_dataframe(self, df: pd.DataFrame, id_col: Optional[str] = None) -> None:
        """
        Ingests a DataFrame with columns ['scene_start', 'scene_end', 'transcript', 'embedding', ...] into Qdrant.
        Args:
            df (pd.DataFrame): DataFrame to ingest. Must have 'embedding' column (list/array).
            id_col (Optional[str]): Optional column to use as point ID. If None, use DataFrame index.
        """
        points = []
        for idx, row in df.iterrows():
            vector = row['embedding']
            payload = {k: v for k, v in row.items() if k != 'embedding'}
            point_id = int(row[id_col]) if id_col and id_col in row else int(idx)
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_vector: List[float], top_k: int = 5, with_payload: bool = True) -> List[Dict[str, Any]]:
        """
        Query Qdrant for the most similar points to the given vector.
        Args:
            query_vector (List[float]): The embedding vector to search with.
            top_k (int): Number of results to return.
            with_payload (bool): Whether to return payloads.
        Returns:
            List of dicts with point IDs, scores, and payloads (if requested).
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=with_payload
        )
        return [
            {
                'id': r.id,
                'score': r.score,
                'payload': r.payload if with_payload else None
            } for r in results
        ]

# Example usage (for testing, not production):
if __name__ == "__main__":
    import numpy as np
    # Example DataFrame
    df = pd.DataFrame([
        {'scene_start': 0.0, 'scene_end': 10.0, 'transcript': 'Hello world', 'embedding': np.random.rand(768).tolist()},
        {'scene_start': 10.0, 'scene_end': 20.0, 'transcript': 'Another scene', 'embedding': np.random.rand(768).tolist()}
    ])
    wrapper = QdrantClientWrapper(collection_name='test_scenes', vector_size=768)
    wrapper.ingest_dataframe(df)
    query_vec = df.iloc[0]['embedding']
    results = wrapper.query(query_vec)
    print(results) 