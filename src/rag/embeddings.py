from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger


def get_huggingface_embeddings(model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    logger.info(f"Loading HuggingFace embeddings (ONNX): {model}")
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"backend": "onnx", "device": "cpu"},
        encode_kwargs={"normalize_embeddings": True})


