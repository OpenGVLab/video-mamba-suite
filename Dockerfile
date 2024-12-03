# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS build


# Set build arguments for UID and GID
ARG UID=1000
ARG GID=1000

WORKDIR /workspace
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt


# Create a group and user with the specified UID and GID
RUN groupadd -g ${GID} usergroup && \
    useradd -m -u ${UID} -g usergroup user


# Pre-final stage
FROM build AS pre

# Set the working directory
WORKDIR /workspace

# Copy the application code
COPY . /workspace

RUN pip install causal-conv1d/

# Final stage
FROM pre AS final

RUN pip install mamba/

# Set environment variables
ENV XDG_CACHE_HOME=/workspace/.cache
ENV HOME=/workspace

EXPOSE 8888