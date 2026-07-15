FROM python:3.11

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m src.data_ingestion.scrapper_who
RUN python -m src.data_ingestion.clean_data
RUN python -m src.rag.pipeline build

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
