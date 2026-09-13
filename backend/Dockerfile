# Build context must be the REPO ROOT (not backend/), e.g.:
#   docker build -f backend/Dockerfile -t nvme-backend .
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY backend/model ./backend/model
# ml/data is needed for /fleet-overview, which reads the CSV directly
COPY ml/data ./ml/data

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
