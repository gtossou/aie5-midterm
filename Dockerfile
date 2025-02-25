# Get a distribution that has uv already installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Add user - this is the user that will run the app
# If you do not set user, the app will run as root (undesirable)
RUN useradd -m -u 1000 user
USER user

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets

# Set the working directory
WORKDIR $HOME/app

# Install system dependencies (curl and build tools)
USER root
RUN apt-get update && apt-get install -y curl build-essential
USER user

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/user/.cargo/bin:${PATH}"

# Copy the app to the container
COPY --chown=user . $HOME/app

# Install the dependencies
# RUN uv sync --frozen
RUN uv sync

# Expose the port
EXPOSE 7860

# Move to the .venv directory (if it exists)
WORKDIR $HOME/app/.venv
# Run the app
CMD ["uv", "run", "chainlit", "run", "../app.py", "--host", "0.0.0.0", "--port", "7860"]