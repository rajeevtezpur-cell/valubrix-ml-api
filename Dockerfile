# Use stable Python version (avoids pandas build errors)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run training script
CMD ["python", "train.py"]
