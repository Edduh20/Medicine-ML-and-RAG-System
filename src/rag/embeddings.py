from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger


def get_huggingface_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    logger.info(f"Loading HuggingFace embeddings: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


