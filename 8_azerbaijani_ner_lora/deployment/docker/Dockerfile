# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set working directory in the container
WORKDIR /app

# Install system dependencies needed for PyYAML (if any) and others
# Although PyYAML is pure Python, some packages might have system dependencies.
# This is a general practice.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# This copies the entire 8_azerbaijani_ner_lora project structure into the container.
# Adjust if only specific parts are needed to reduce image size.
COPY . /app

# Ensure the project root is on the Python path
ENV PYTHONPATH=/app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
# This assumes uvicorn is installed via requirements.txt
CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
