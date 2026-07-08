FROM nvcr.io/nvidia/pytorch:25.02-py3

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

#RUN pip3 install --no-cache-dir --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install remaining python libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Force Hugging Face to save models inside the mounted project volume
ENV HF_HOME=/workspace/.hf_cache

# Default command to verify setup
CMD ["python3", "-c", "import torch; print('Container Active. CUDA Available:', torch.cuda.is_available())"]