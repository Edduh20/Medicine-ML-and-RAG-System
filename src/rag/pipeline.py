from dotenv import load_dotenv
from src.rag.document_loader import load_documents
from src.rag.chunk import chunk_documents
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import build_vector_store, load_vector_store
from src.rag.retriever import get_retriever
from src.rag.chain import get_llms, build_rag_chain_with_context
from src.utils.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_decompose_prompt = ChatPromptTemplate.from_template(
    "Split the following question into individual sub-questions, one per distinct topic or disease. "
    "Return them as a comma-separated list. If there is only one topic, return it as-is.\n\n"
    "Question: {question}\n\nSub-questions:"
)


def decompose(question: str, llm) -> list[str]:
    chain = _decompose_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return [q.strip() for q in result.split(",")]



_state = {}

def _get_embeddings():
    if "embeddings" not in _state:
        logger.info("Loading embeddings...")
        _state["embeddings"] = get_embeddings() 
    return _state["embeddings"]

def _get_retriever():
    if "retriever" not in _state:
        logger.info("Connecting to Qdrant and building retriever...")
        vs = load_vector_store(_get_embeddings())
        _state["retriever"] = get_retriever(vs, k=8)
        logger.info("Vector store and retriever ready.")
    return _state["retriever"]

_cache = {
    "chain": None,
    "answers": {}
}


def build_index(doc_path: str):
    documents = load_documents(doc_path)
    chunks = chunk_documents(documents)
    build_vector_store(chunks, _get_embeddings())  
    logger.info("Knowledge base indexed in Qdrant.")


def answer_question(question: str) -> dict:
    global _cache

    cleaned_question = question.lower().strip()

    if cleaned_question in _cache["answers"]:
        return _cache["answers"][cleaned_question]

    providers = ["gemini", "groq", "cohere", "mistral"]

    for provider in providers:
        try:
            logger.info(f"Trying {provider}...")
            llm = get_llms(provider)

            sub_questions = decompose(question, llm)
            logger.info(f"Decomposed into: {sub_questions}")

            all_docs = []
            for sub_q in sub_questions:
                docs = _get_retriever().invoke(sub_q)           
                all_docs.extend(docs)

            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = doc.page_content[:150]
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)

            context = "\n\n".join(doc.page_content for doc in unique_docs[:30])

            chain = build_rag_chain_with_context(llm, context, unique_docs[:30])
            result = chain.invoke({"question": question})

            response = {
                "answer": result["result"],
                "sources": [doc.metadata for doc in result["source_documents"]],
            }

            _cache["chain"] = chain
            _cache["answers"][cleaned_question] = response
            return response

        except Exception as e:
            logger.warning(f"{provider} failed: {e}")
            continue

    raise RuntimeError("All LLM providers failed.")


if __name__ == "__main__":
    pass   