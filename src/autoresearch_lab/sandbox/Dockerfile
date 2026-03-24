FROM python:3.12-slim

# Install Node.js 22
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code@2.1.81

# Install autoresearch-lab (provides the `arl` CLI).
# By default installs from PyPI. When building from source, the CLI passes
# --build-arg ARL_INSTALL_SPEC=/tmp/autoresearch-lab and sets the build
# context to the package root so COPY picks up the source.
COPY . /tmp/autoresearch-lab
ARG ARL_INSTALL_SPEC=autoresearch-lab
RUN pip install --no-cache-dir ${ARL_INSTALL_SPEC} && rm -rf /tmp/autoresearch-lab

# Create non-root user
RUN useradd -m -s /bin/bash agent
USER agent

WORKDIR /workspace

ENTRYPOINT ["claude"]
