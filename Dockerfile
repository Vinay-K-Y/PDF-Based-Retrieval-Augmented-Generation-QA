# 1. Start with a clean, lightweight Python OS
FROM python:3.11-slim

# 2. Set our working folder inside the container
WORKDIR /app

# 3. Install the system-level Tesseract software required for your OCR fallback
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your newly cleaned requirements file
COPY requirements.txt .

# 5. NEW: Upgrade pip so it handles downloads better
RUN pip install --no-cache-dir --upgrade pip

# 6. NEW: Increase the timeout limit for slower network downloads
ENV PIP_DEFAULT_TIMEOUT=100

# 7. Install the lightweight CPU-only version of PyTorch first!
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 8. Install the rest of the ML and web dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 9. Copy the rest of your project files (app.py, HTML, Python scripts)
COPY . .

# 10. Expose port 5000 so we can access Flask
EXPOSE 5000

# 11. The command to start your DocuMind API
CMD ["python", "app.py"]