from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.utils.logger import logger


def get_embeddings(model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    logger.info(f"Loading FastEmbed embeddings (ONNX, no torch): {model}")
    return FastEmbedEmbeddings(
        model_name=model,
        cache_dir="models/fastembed_cache",
    )