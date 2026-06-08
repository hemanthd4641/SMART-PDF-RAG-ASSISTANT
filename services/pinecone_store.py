from typing import List, Dict, Any, Optional
import time
from pinecone import Pinecone, ServerlessSpec
from utils.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from utils.helpers import get_logger

logger = get_logger("pinecone")

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384
# Pinecone free-tier serverless cloud/region
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

class PineconeStore:
    """Pinecone vector database store interface using the Pinecone Python SDK."""

    def __init__(self):
        self.api_key = PINECONE_API_KEY
        self.index_name = PINECONE_INDEX_NAME
        self.client = None
        self.index = None

    def connect(self) -> None:
        """
        Establishes connection to Pinecone client and index.
        Auto-creates the index with the correct spec if it does not exist yet.
        """
        if self.index is not None:
            return

        if not self.api_key:
            logger.error("PINECONE_API_KEY is missing from configurations.")
            raise ValueError("Pinecone API Key is not configured.")

        try:
            logger.info("Initializing Pinecone client...")
            self.client = Pinecone(api_key=self.api_key)

            existing_indexes = [idx.name for idx in self.client.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(
                    f"Pinecone index '{self.index_name}' not found. "
                    f"Auto-creating with dim={EMBEDDING_DIMENSION}, metric=cosine, "
                    f"cloud={PINECONE_CLOUD}, region={PINECONE_REGION}..."
                )
                self.client.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )

                # Wait until the index is ready before connecting
                logger.info(f"Waiting for index '{self.index_name}' to become ready...")
                for attempt in range(30):  # up to ~60 seconds
                    status = self.client.describe_index(self.index_name).status
                    if status.get("ready", False):
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Timed out waiting for Pinecone index '{self.index_name}' to become ready."
                    )

                logger.info(f"Pinecone index '{self.index_name}' created and ready.")

            self.index = self.client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: '{self.index_name}' successfully.")

        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise RuntimeError(f"Pinecone connection failure: {e}")

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Upserts vectors into the Pinecone index.
        
        Args:
            vectors: List of dictionaries matching the format:
                {
                    "id": str,                  # unique chunk ID
                    "values": list of floats,   # dense embedding vector
                    "metadata": {               # metadata to index
                        "document_name": str,
                        "page_number": int,
                        "text": str
                    }
                }
        """
        self.connect()
        logger.info(f"Preparing to upsert {len(vectors)} vectors to Pinecone...")
        
        try:
            # Reformat to match Pinecone client expectations: tuples of (id, values, metadata)
            pinecone_payload = []
            for vec in vectors:
                chunk_id = vec.get("id")
                values = vec.get("values")
                metadata = vec.get("metadata", {})
                
                # Validation checks
                if not chunk_id or not values:
                    logger.warning(f"Skipping malformed vector item: {vec}")
                    continue
                    
                pinecone_payload.append((chunk_id, values, metadata))

            if pinecone_payload:
                # Upsert payload in batches
                self.index.upsert(vectors=pinecone_payload)
                logger.info(f"Successfully upserted {len(pinecone_payload)} vectors to Pinecone.")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {e}")
            raise RuntimeError(f"Pinecone upsert failure: {e}")

    def delete_document_vectors(self, document_name: str) -> None:
        """
        Deletes all chunks associated with a specific document name from the Pinecone index
        using metadata filtering.
        
        Args:
            document_name: Filename string of the document to clean up.
        """
        self.connect()
        logger.info(f"Deleting all vectors associated with document: '{document_name}'")
        
        try:
            # Metadata filter deletion syntax for Pinecone
            self.index.delete(filter={"document_name": {"$eq": document_name}})
            logger.info(f"Successfully deleted vectors for document '{document_name}' from Pinecone.")
        except Exception as e:
            logger.error(f"Failed to delete document vectors from Pinecone: {e}")
            raise RuntimeError(f"Pinecone deletion failure: {e}")

    def query_vectors(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries Pinecone index for top_k similar vector matches.
        
        Args:
            query_vector: The dense embedding float list of the query.
            top_k: Number of results to return.
            filter_dict: Metadata filtering criteria (e.g. {"document_name": "filename.pdf"}).
            
        Returns:
            List of matching records with similarity scores and metadata.
        """
        self.connect()
        logger.info(f"Querying top_{top_k} similar vectors from Pinecone (filter={filter_dict})...")
        
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            return results
        except Exception as e:
            logger.error(f"Failed to query vectors from Pinecone: {e}")
            raise RuntimeError(f"Pinecone query failure: {e}")
