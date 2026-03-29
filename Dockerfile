FROM python:3.12

WORKDIR /app

# install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# run app
CMD ["python", "app.py"]
