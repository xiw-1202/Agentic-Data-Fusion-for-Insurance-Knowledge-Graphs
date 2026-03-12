#!/usr/bin/env bash
# =============================================================================
# cluster_setup.sh — One-time environment setup on Emory Turing GPU cluster
#
# Usage (run once after SSH-ing in):
#   bash /local/scratch/$USER/project/scripts/cluster_setup.sh
#
# What it does:
#   1. Creates scratch directories (models, logs, .cache)
#   2. Installs Ollama with GPU support
#   3. Pulls llama3.1:70b (4-bit quantized, ~40 GB)
#   4. Creates Python venv and installs project dependencies
#   5. Redirects HuggingFace/pip/wandb cache to scratch
#
# Pre-requisite: copy project to scratch first:
#   scp -r ~/path/to/CS584_AI_Capstone turinglogin:/local/scratch/$USER/project
# =============================================================================

set -e
NETID=$USER
SCRATCH=/local/scratch/$NETID
PROJECT=$SCRATCH/project

echo "=== CS584 Turing Cluster Setup ==="
echo "NETID:   $NETID"
echo "SCRATCH: $SCRATCH"
echo "PROJECT: $PROJECT"
echo ""

# ---------------------------------------------------------------------------
# 1. Scratch directories
# ---------------------------------------------------------------------------
mkdir -p $SCRATCH/{models,logs,.cache,venv}
echo "[1/5] Scratch directories created"

# ---------------------------------------------------------------------------
# 2. Install Ollama (GPU-enabled) — no sudo required, installs to scratch
# ---------------------------------------------------------------------------
mkdir -p $SCRATCH/bin
export PATH="$SCRATCH/bin:$PATH"

if ! command -v ollama &>/dev/null; then
    echo "[2/5] Downloading Ollama (linux-amd64)..."
    curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
        -o /tmp/ollama.tar.zst

    # Decompress .tar.zst — archive layout is bin/ollama + lib/ollama/
    # so extract to $SCRATCH (not $SCRATCH/bin) to get $SCRATCH/bin/ollama
    if command -v zstd &>/dev/null; then
        zstd -d -c /tmp/ollama.tar.zst | tar -xC $SCRATCH/
    else
        echo "  zstd not in PATH — using Python zstandard to decompress..."
        pip3 install --target=$SCRATCH/.cache/pyutils zstandard -q
        PYTHONPATH=$SCRATCH/.cache/pyutils python3 << PYEOF
import zstandard, tarfile
with open('/tmp/ollama.tar.zst', 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        # mode='r|' = pipe/stream mode — no seeking required (zstd streams are non-seekable)
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            tar.extractall('$SCRATCH')
print("  Extraction complete")
PYEOF
    fi
    rm -f /tmp/ollama.tar.zst
    chmod +x $SCRATCH/bin/ollama
    echo "[2/5] Ollama installed: $($SCRATCH/bin/ollama --version)"
else
    echo "[2/5] Ollama already installed: $(ollama --version)"
fi

# Point Ollama model storage to scratch (not home)
export OLLAMA_MODELS=$SCRATCH/models

# ---------------------------------------------------------------------------
# 3. Pull llama3.1:70b (4-bit, ~40 GB — fits in one 48 GB GPU)
# ---------------------------------------------------------------------------
echo "[3/5] Pulling llama3.1:70b (this will take 20-40 min on first run)..."
ollama serve &>/dev/null &
OLLAMA_PID=$!
sleep 5   # wait for server to start

ollama pull llama3.1:70b
echo "  llama3.1:70b ready"

# Optional: also pull 8b for quick comparison runs
# ollama pull llama3.1:8b

kill $OLLAMA_PID 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Python virtual environment
# ---------------------------------------------------------------------------
echo "[4/5] Creating Python venv..."
# Use virtualenv per cluster instructions; fall back to python3 -m venv if not available
if command -v virtualenv &>/dev/null; then
    virtualenv -p python3 $SCRATCH/venv
else
    python3 -m venv $SCRATCH/venv
fi
source $SCRATCH/venv/bin/activate

# Redirect caches to scratch (required by cluster policy)
# Also add $SCRATCH/bin to PATH so Slurm jobs can find the ollama binary
echo "export PATH=$SCRATCH/bin:\$PATH" >> $SCRATCH/venv/bin/activate
echo "export XDG_CACHE_HOME=$SCRATCH/.cache" >> $SCRATCH/venv/bin/activate
echo "export HF_HOME=$SCRATCH/.cache/huggingface" >> $SCRATCH/venv/bin/activate
echo "export OLLAMA_MODELS=$SCRATCH/models" >> $SCRATCH/venv/bin/activate
echo "export PIP_CACHE_DIR=$SCRATCH/.cache/pip" >> $SCRATCH/venv/bin/activate

# Re-source to pick up cache exports
source $SCRATCH/venv/bin/activate

pip install --upgrade pip --quiet
pip install -r $PROJECT/requirements.txt --quiet
echo "  Python dependencies installed"

# ---------------------------------------------------------------------------
# 5. Verify GPU visibility
# ---------------------------------------------------------------------------
echo "[5/5] GPU check:"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "  (torch not installed — that's fine, Ollama uses GPU directly)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | awk '{print "  GPU: "$0}'

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env file:  scp ~/.env_cs584 turinglogin:$PROJECT/.env"
echo "  2. Submit job:      sbatch $PROJECT/scripts/slurm_zone2_zone3.sh"
echo "  3. Monitor:         squeue -u $NETID"
echo "  4. Logs:            tail -f $SCRATCH/logs/<jobid>.out"
