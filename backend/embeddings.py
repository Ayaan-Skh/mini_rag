from sentence_transformers import SentenceTransformer
from typing import List
import logging

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings of the docs and user input
    """
    def __init__(self,model_name:str=None):
        """
        Initialize embeddings generator
        """
        self.model_name=model_name or settings.embedding_model
        try:
            self.model=SentenceTransformer(self.model_name)
            logger.info(f"Model: {self.model_name}  loaded successfully...")
        except Exception as e:
            logger.error(f"Failed to load embeddings model:{e}")    
            raise
        
    def embed_text(self,text:str)->List[float]:
        try:
            embeddings=self.model.encode(text,convert_to_numpy=True)
            logging.info("Embeddings generated")
            return embeddings.tolist()    
        except Exception as e:
            logging.error(f"Error generating embedding:{e}")
            raise
        
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

            