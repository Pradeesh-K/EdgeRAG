FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install CPU-only torch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0

# Then install rest of requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
