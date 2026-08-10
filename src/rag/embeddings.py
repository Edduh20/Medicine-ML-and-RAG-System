import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.utils.logger import logger


def get_huggingface_embeddings(model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    logger.info(f"Loading FastEmbed embeddings: {model}")
    return FastEmbedEmbeddings(model_name=model)


