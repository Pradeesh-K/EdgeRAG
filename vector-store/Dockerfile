# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .

# Expose port # reachable by other applications
EXPOSE 5002 

# Start Flask
CMD ["python", "app.py"]
