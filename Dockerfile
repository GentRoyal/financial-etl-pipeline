FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app folder (including main.py and scripts)
COPY app/ ./app

# Copy your environment file into the container
COPY v.env ./v.env

ENV ENVIRONMENT=docker

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]