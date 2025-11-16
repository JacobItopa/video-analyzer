FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Create app directory
WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the server
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "fastapi_app:app"]
