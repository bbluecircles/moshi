# Use NVIDIA CUDA 12.8.0 devel image with Ubuntu 22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Phoenix

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
# Assuming you have a requirements.txt file in the moshi directory
RUN pip install --no-cache-dir -r requirements.txt

# Install Moshi and gradio
RUN pip install --no-cache-dir moshi gradio fastapi

# Install Ngrok
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list \
    && apt-get update \
    && apt-get install -y ngrok \
    && rm -rf /var/lib/apt/lists/*

# Expose the port used by the server
EXPOSE 8000

# Set environment variable for the model (with a default value)
ENV HF_REPO=kyutai/moshiko-pytorch-bf16

# Use entrypoint to start both FastAPI and Ngrok
ENTRYPOINT ["./entrypoint.sh"]
