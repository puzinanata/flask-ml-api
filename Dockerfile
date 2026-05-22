FROM python:3.12-slim

WORKDIR /app

# Install production dependencies first
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy app files
COPY . .

# Railway provides PORT automatically
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000}