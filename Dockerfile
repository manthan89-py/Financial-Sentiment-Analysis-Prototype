# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install pip and uv
RUN pip install --no-cache-dir uv

# install dependencies
RUN uv pip install --requirement pyproject.toml --system

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Expose the port
EXPOSE 8000

# Set entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]