from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.prediction.predict import predict
from src.rag.pipeline import answer_question

app = FastAPI(title="DiagnosAI API", version="0.2.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "ML + RAG API is running"}


class PredictionRequest(BaseModel):
    symptoms: str


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    try:
        return predict(request.symptoms)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown symptom: {e}")


class RAGRequest(BaseModel):
    question: str


@app.post("/answer_question")
def answer_question_endpoint(request: RAGRequest):
    try:
        return answer_question(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))