from typing import List
from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL_NAME
from utils.helpers import get_logger, time_it

logger = get_logger("embeddings")

class EmbeddingService:
    """Embedding generation service using sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None  # Lazy load to avoid application startup lag

    def load_model(self) -> None:
        """Loads the SentenceTransformer model weights into memory if not already loaded."""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model weights: '{self.model_name}'")
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize sentence-transformer '{self.model_name}': {e}")
                raise RuntimeError(f"Embedding model initialization failure: {e}")

    @time_it
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of input strings.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size to process.
            
        Returns:
            List of embedding vectors (list of floats).
            
        Raises:
            RuntimeError: If generation fails.
        """
        if not texts:
            return []
            
        # Ensure the model is loaded
        self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        
        try:
            # sentence-transformers natively handles batching inside model.encode
            # convert_to_numpy ensures we get numpy arrays back
            embeddings_ndarray = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert numpy array (e.g. of shape N x 384) to lists of python floats
            return embeddings_ndarray.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failure: {e}")
