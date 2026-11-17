from langchain_community.vectorstores import FAISS
from src.utils.logger import logger


def get_retriever(vector_store: FAISS, k: int = 8):
    logger.info(f"Creating retriever (top_k={k}, MMR)")
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5, "score_threshold": 0.35}
    )