FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/figures/regime outputs/figures/validation \
    outputs/forecasts outputs/logs outputs/qa_report \
    data/raw data/processed

EXPOSE 8501 8000

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
