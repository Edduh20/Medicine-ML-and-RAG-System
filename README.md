# 🩺 Medicine Recommendation and Medical RAG System

An AI-powered healthcare application that combines machine learning and Retrieval-Augmented Generation (RAG) to provide disease prediction, medicine recommendations, and evidence-based medical information.

## Features

- Disease prediction from symptoms
- Medicine recommendation
- WHO knowledge base integration
- Retrieval-Augmented Generation (RAG)
- Hugging Face embeddings
- FAISS vector database
- Gemini-powered medical assistant
- FastAPI backend
- Streamlit frontend

## Structure
| Folder | Purpose |
|--------|---------|
| `data/` | Raw, processed, external, interim datasets |
| `notebooks/` | EDA and experimentation |
| `src/` | Modular pipeline source code |
| `artifacts/` | Encoders, scalers, train/test splits |
| `models/` | Trained model files |
| `reports/` | Figures, charts, summaries |
| `api/` | FastAPI app for serving predictions |
| `tests/` | Unit and integration tests |
| `logs/` | Runtime logs |
| `config/` | YAML config and hyperparameters |
| `deployment/` | Docker, CI/CD configs |

## Setup
```bash
pip install -r requirements.txt
python setup.py develop
```

## Run API
```bash
uvicorn api.app:app --reload
```

## Run Tests
```bash
pytest tests/
```
