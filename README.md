# 🩺 DiagnosAI — Medical Prediction & RAG-Powered Knowledge Assistant

An end-to-end healthcare application that combines a supervised ML pipeline with Retrieval-Augmented Generation (RAG) to predict likely diseases from symptoms, recommend medicines and answer follow-up medical questions using verified WHO knowledge, not just LLM guesswork.

Built from first principles as a personal deep-dive into production ML + LLM systems, rather than a tutorial clone.

---

## 🔗 Live Demo

**[Try DiagnosAI here](https://diagnosaimedicine-ml-and-rag-system-2.streamlit.app/)**

---

## How It Works

The app has two independent entry points: a symptom-based disease predictor and a chat interface for open-ended medical questions grounded in WHO documentation.

**1. Symptom Prediction**
```
Symptoms (user input)
      │
      ▼
┌─────────────────────┐
│  ML Prediction       │  Scikit-learn model trained on symptom-disease data
│  (FastAPI endpoint)  │  → predicted disease + confidence
└─────────────────────┘
      │
      ▼
Streamlit UI → predicted disease + recommended precautions/medicine
```

**2. Chat (RAG)**
```
User question (free text)
      │
      ▼
┌──────────────────────┐
│  RAG Layer            │  LangChain orchestration
│  • Qdrant vector store│  → semantic search over WHO fact sheets
│  • HuggingFace         │  → embeddings for retrieval
│    embeddings          │
│  • Response cache      │  → skips repeat LLM calls for identical queries
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  Multi-provider LLM    │  Tries providers in order (Gemini → Groq →
│  fallback chain        │  Cohere → Mistral), falling through on
│                        │  rate limits/failures for resilience
└──────────────────────┘
      │
      ▼
Streamlit UI → evidence-based answer grounded in WHO documentation
```

These two flows are independent: The ML model doesn't feed the RAG chain and vice versa. A user can predict a disease from symptoms, ask a free-form medical question, or both, in the same session.

The RAG chain is provider-agnostic: If the primary LLM (Gemini) hits a rate limit or fails, the pipeline automatically retries with the next configured provider, improving reliability without user-facing downtime.

---

## Features

- **Disease prediction** from reported symptoms using a trained Scikit-learn classifier
- **Medicine & precaution recommendations** tied to predicted conditions
- **RAG-grounded Q&A** over WHO fact sheets reduces hallucination on medical facts
- **Response caching** to avoid redundant LLM calls for repeated queries
- **Multi-provider LLM fallback** (Gemini → Groq → Cohere → Mistral) for resilience against rate limits/outages
- **FastAPI backend** serving predictions and RAG responses
- **Streamlit frontend** for interactive use

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn, Pandas, NumPy |
| RAG orchestration | LangChain |
| Embeddings | HuggingFace |
| Vector store | Qdrant |
| LLM | Gemini, Groq, Cohere, Mistral (fallback chain) |
| API | FastAPI |
| Frontend | Streamlit |
---

## Project Structure

| Folder | Purpose |
|---|---|
| `data/` | Raw, processed, and interim datasets |
| `notebooks/` | EDA and experimentation |
| `src/` | Modular pipeline source code (data prep → training → evaluation) |
| `artifacts/` | Encoders, scalers, train/test splits |
| `models/` | Trained model files |
| `api/` | FastAPI app serving predictions and RAG responses |
| `app/` | Streamlit frontend, connected to the API |
| `config/` | YAML config and hyperparameters |
| `deployment/` | Docker and CI/CD configs |
| `tests/` | Unit and integration tests |
| `.devcontainer/` | Containerized dev environment config |

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/Edduh20/Medicine-ML-and-RAG-System.git
cd Medicine-ML-and-RAG-System

# Create environment
conda create -n medrag python=3.11
conda activate medrag

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
GROQ_API_KEY=your_api_key_here
COHERE_API_KEY=your_api_key_here
MISTRAL_API_KEY=your_api_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key_here
```

### Build the knowledge base index

Before running the RAG layer, embed the WHO fact sheets and index them into Qdrant:

```bash
python -c "from pipeline import build_index; build_index('path/to/who_docs')"
```

---

## Running the App

**API (FastAPI):**
```bash
uvicorn api.app:app --reload
```

**Frontend (Streamlit):**
```bash
streamlit run app/main.py
```

**Tests:**
```bash
pytest tests/
```

> Note: scripts under `src/` expect to be run from within that directory due to relative path handling.

---