from sentence_transformers import SentenceTransformer
from typing import List
import logging

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings of the docs and user input
    """
    def __init__(self, model_name: str = None):
        """Initialize the embedding generator."""
        self.model_name = model_name or settings.embedding_model
        self.model = None  # Don't load yet
        logger.info(f"Embedding generator created (model will load on first use)")

    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        self._ensure_model_loaded()
        
    def embed_batch(self,texts:List[str],batch_size:int)->List[float]:
        """
        Generate embeddings for multiple texts
        """    
        try:    
            embeddings=self.model.encode(texts, 
                                        batch_size=batch_size,
                                        convert_to_numpy=True,
                                        show_progress_bar=True)
            logging.info(f"Generated embeddings for texts of len {len(texts)}")
            # Convert numpy array into list
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error generating embedidngs:{e}")
            raise
   
embedding_generator=None        
def get_embedding_generator()->EmbeddingGenerator:
    """
    To get generator for once and reuse it.
    """        
    global embedding_generator
    if embedding_generator is None:
        embedding_generator=EmbeddingGenerator()
    return embedding_generator

            