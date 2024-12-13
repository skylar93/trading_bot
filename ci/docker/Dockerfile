# Base image with CUDA and Ubuntu
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file to leverage Docker caching
COPY ../../requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ../../ .

# Set the default command to run the training script
CMD ["python3", "training/train.py"]

