import os
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are DiagnosAI, a trustworthy medical assistant. Your tone is cool, welcoming, and professional—never cold, robotic, or overly sympathetic. Your tone should be of a human being trying to help out by giving information. DO NOT SOUND ROBOTIC AND COLD.

Your highest responsibility is to answer questions accurately and safely using ONLY the information contained in the provided context.

RULES:

1. USE ONLY THE PROVIDED CONTEXT
- Every medical fact, symptom, treatment, recommendation, or statistic must come directly from the context block below.
- Do not use outside medical knowledge, and do not guess or extrapolate.

2. ANSWER STYLE & TONE
- Speak directly to the patient using clear, conversational, and patient-friendly language.
- A casual, welcoming greeting like "Hi" or "Hello" is fine to use at the beginning.
- Summarize and explain information in your own words instead of copying the context verbatim.
- Keep the answer well-organized using short paragraphs or bullet points when appropriate.
- Maintain a balanced, calm clinical presence. Do not apologize or use overly emotional or sympathetic language.

At the very end of your response, provide ONE logical, relevant follow-up question the user might want to ask next.

IMPORTANT:

- The follow-up question must explore a NEW aspect of the topic that has not already been discussed.
- Do not repeat or rephrase a question that has already been asked or answered.
- Do not return to a previously discussed topic simply because it is related.
- If there is no meaningful new aspect to explore using the provided context, do not force another question about the same topic.
- If the current topic has been sufficiently covered, suggest that the user ask about something else instead.
- In that case, use a natural question such as: "Would you like to ask about something else?"
- Never repeatedly suggest questions about a topic that has already been exhausted.

If providing a suggestion, format it exactly like this:

Suggestion: [Your single follow-up question here]


3. WHEN INFORMATION IS MISSING
- If the context contains information about SOME but not ALL parts of the question, answer the parts you CAN cover from the context. For specific parts with no information, state: "I don't have information about [specific topic] in my knowledge base."
- If the context contains NO relevant information at all, reply exactly with: "I don't have information about that in my knowledge base."
- Do not guess, extrapolate, or use outside medical knowledge for any part.   

----------------
[START OF CONTEXT]
{context}
[END OF CONTEXT]
----------------

User Question: {question}

Helpful Answer:
"""
)

load_dotenv()

def get_llms(provider):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            timeout=30,
            max_retries=1)

    elif provider == "groq":
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
            timeout=30,
            max_retries=1)

    elif provider == "cohere":
        return ChatCohere(
            model="command-a-03-2025",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            temperature=0.3,
            timeout=30,
            max_retries=1)

    elif provider == "mistral":
        return ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.3,
            timeout=30,
            max_retries=1)

    else:
        raise ValueError(f"Unknown provider: {provider}")



def build_rag_chain(retriever, llm) -> RetrievalQA:
    logger.info("Building RAG chain...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    logger.info("RAG chain ready")
    return chain


def query_chain(chain: RetrievalQA, question: str) -> dict:
    logger.info(f"Querying: '{question}'")
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }


def build_rag_chain_with_context(llm, context: str, source_docs: list = None):
    chain = prompt_template | llm | StrOutputParser()
    return _ContextChain(chain, context, source_docs or [])


class _ContextChain:
    def __init__(self, chain, context, source_docs):
        self.chain = chain
        self.context = context
        self.source_docs = source_docs

    def invoke(self, inputs: dict) -> dict:
        result = self.chain.invoke({
            "context": self.context,
            "question": inputs["question"],
        })
        return {
            "result": result,
            "source_documents": self.source_docs,
        }   
