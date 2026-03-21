#!/usr/bin/env bash
# =============================================================================
# cluster_setup_v2.sh — One-time environment setup on Emory Turing GPU cluster
#
# MODEL RECOMMENDATION: qwen2.5:72b
#   - 2× 48 GB GPUs = 96 GB total VRAM
#   - qwen2.5:72b (Q4_K_M) ≈ 43 GB → fits on 1 GPU, or split across 2 for speed
#   - Superior structured JSON output vs llama3.1:70b
#   - Better instruction following for multi-triple extraction
#   - Also pulls llama3.3:70b as backup (newer than 3.1, better instruction following)
#
# Usage (run once after SSH-ing in):
#   bash /local/scratch/$USER/project/scripts/cluster_setup_v2.sh
#
# Pre-requisite: copy project to scratch first:
#   bash scripts/sync_to_cluster.sh xwa2284
# =============================================================================

set -e
NETID=${USER:-xwa2284}
SCRATCH=/local/scratch/$NETID
PROJECT=$SCRATCH/project

echo "=== CS584 Turing Cluster Setup v2 (2× 48GB GPUs) ==="
echo "NETID:   $NETID"
echo "SCRATCH: $SCRATCH"
echo "PROJECT: $PROJECT"
echo ""

# ---------------------------------------------------------------------------
# 1. Scratch directories
# ---------------------------------------------------------------------------
mkdir -p $SCRATCH/{models,logs,.cache,venv,bin,lib}
echo "[1/6] Scratch directories created"

# ---------------------------------------------------------------------------
# 2. Install Ollama (GPU-enabled) — no sudo required, installs to scratch
# ---------------------------------------------------------------------------
export PATH="$SCRATCH/bin:$PATH"

if [ ! -f "$SCRATCH/bin/ollama" ]; then
    echo "[2/6] Downloading Ollama (linux-amd64)..."
    curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
        -o /tmp/ollama.tar.zst

    if command -v zstd &>/dev/null; then
        zstd -d -c /tmp/ollama.tar.zst | tar -xC $SCRATCH/
    else
        echo "  zstd not in PATH — using Python zstandard to decompress..."
        pip3 install --target=$SCRATCH/.cache/pyutils zstandard -q
        PYTHONPATH=$SCRATCH/.cache/pyutils python3 -c "
import zstandard, tarfile
with open('/tmp/ollama.tar.zst', 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            tar.extractall('$SCRATCH')
print('  Extraction complete')
"
    fi
    rm -f /tmp/ollama.tar.zst
    chmod +x $SCRATCH/bin/ollama
    echo "[2/6] Ollama installed: $($SCRATCH/bin/ollama --version)"
else
    echo "[2/6] Ollama already installed: $($SCRATCH/bin/ollama --version)"
fi

# Point Ollama model storage to scratch (not home)
export OLLAMA_MODELS=$SCRATCH/models

# ---------------------------------------------------------------------------
# 3. Pull models — primary: qwen2.5:72b, backup: llama3.3:70b
# ---------------------------------------------------------------------------
echo "[3/6] Starting Ollama to pull models..."
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}
ollama serve &>/dev/null &
OLLAMA_PID=$!

# Wait for server
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  Ollama ready (${i}s)"
        break
    fi
    sleep 1
done

echo "  Pulling qwen2.5:72b (~43 GB, Q4_K_M quantization)..."
echo "  This will take 20-40 min on first run."
ollama pull qwen2.5:72b
echo "  ✅ qwen2.5:72b ready"

echo "  Pulling llama3.3:70b (~43 GB, backup model)..."
ollama pull llama3.3:70b
echo "  ✅ llama3.3:70b ready"

# Also keep 8b for quick debugging runs
echo "  Pulling llama3.1:8b (~4.7 GB, debug model)..."
ollama pull llama3.1:8b
echo "  ✅ llama3.1:8b ready"

kill $OLLAMA_PID 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Python virtual environment
# ---------------------------------------------------------------------------
echo "[4/6] Creating Python venv..."
if [ ! -d "$SCRATCH/venv/bin" ]; then
    python3 -m venv $SCRATCH/venv
fi
source $SCRATCH/venv/bin/activate

# Redirect caches to scratch (required by cluster policy)
# Check if already configured to avoid duplicates
if ! grep -q "XDG_CACHE_HOME" $SCRATCH/venv/bin/activate 2>/dev/null; then
    cat >> $SCRATCH/venv/bin/activate << ENVEOF

# === CS584 cluster environment ===
export PATH=$SCRATCH/bin:\$PATH
export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_HOME=$SCRATCH/.cache/huggingface
export OLLAMA_MODELS=$SCRATCH/models
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:\${LD_LIBRARY_PATH:-}
ENVEOF
fi

# Re-source to pick up exports
source $SCRATCH/venv/bin/activate

pip install --upgrade pip --quiet
pip install -r $PROJECT/requirements.txt --quiet
echo "  Python dependencies installed"

# ---------------------------------------------------------------------------
# 5. Verify GPU visibility
# ---------------------------------------------------------------------------
echo "[5/6] GPU check:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
    echo "  GPU: $line"
done
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "  Total GPUs visible: $GPU_COUNT"

python3 -c "
import torch
print(f'  PyTorch CUDA: {torch.cuda.is_available()}')
print(f'  PyTorch GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
" 2>/dev/null || echo "  (torch GPU check skipped)"

# ---------------------------------------------------------------------------
# 6. Quick model test
# ---------------------------------------------------------------------------
echo "[6/6] Quick model test (qwen2.5:72b)..."
export OLLAMA_CONTEXT_LENGTH=4096
ollama serve &>/dev/null &
OLLAMA_PID=$!
sleep 5

timeout 120 ollama run qwen2.5:72b 'Reply with a JSON array: [{"entity": "test", "type": "example"}]' --nowordwrap 2>&1 | head -10 || true

kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=== Setup v2 complete! ==="
echo ""
echo "Models available:"
echo "  qwen2.5:72b    — PRIMARY (best structured JSON, knowledge extraction)"
echo "  llama3.3:70b   — BACKUP (strong instruction following)"
echo "  llama3.1:8b    — DEBUG (fast iteration)"
echo ""
echo "Next steps:"
echo "  1. Copy .env:   scp -J xwa2284@lab0z.mathcs.emory.edu .env xwa2284@turinglogin.mathcs.emory.edu:$SCRATCH/project/.env"
echo "  2. Submit job:   sbatch $PROJECT/scripts/slurm_pipeline_v2.sh"
echo "  3. Monitor:      squeue -u $NETID"
echo "  4. Logs:         tail -f $SCRATCH/logs/<jobid>.out"
