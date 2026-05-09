FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/text storage/faiss eval/predictions eval/ground_truth

# Expose ports
EXPOSE 8000 8501

# Use full path for commands
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 & python -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]