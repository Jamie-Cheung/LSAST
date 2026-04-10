#!/bin/bash

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ========================
# CONFIG
# ========================
RUN_NAME="${1:-lsast_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${2:-.}"
GPU_ID="${3:-0}"

# Directories
BASE_DIR="$(pwd)"
PROJECT_DIR="$BASE_DIR"
DATA_DIR="$BASE_DIR/data"
MODEL_DIR="$BASE_DIR/models/sd1.5"
LOG_DIR="$BASE_DIR/logs"
OUTPUT_DIR="$BASE_DIR/outputs/$RUN_NAME"

# Google Drive data source
GDRIVE_FOLDER_ID="1rNGGW8pHQZmCNxL1QyAC_ToswUtpWsjP"
GDRIVE_URL="https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}?usp=drive_link"
SKIP_DOWNLOAD=false
for arg in "$@"; do
    [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=true
done

# Stable Diffusion 1.5 model (NEW URL - sd-legacy)
HF_REPO_SD="sd-legacy/stable-diffusion-v1-5"
MODEL_NAME="v1-5-pruned.ckpt"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# Alternative: download directly from HuggingFace
SD_DOWNLOAD_URL="https://huggingface.co/sd-legacy/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"

# HuggingFace upload repo (optional)
HF_UPLOAD_REPO="${HF_UPLOAD_REPO:-}"

CONFIG="configs/stable-diffusion/v1-finetune.yaml"

HF_TOKEN=${HF_TOKEN:-""}

# ========================
# SETUP
# ========================
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$LOG_DIR" "$OUTPUT_DIR"

cd "$PROJECT_DIR" || exit 1

# Verify config file
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Setup complete${NC}"

# ========================
# CONDA SETUP & ENVIRONMENT
# ========================
CONDA_ENV_NAME="ldm"
CONDA_ENV_FILE="$PROJECT_DIR/environment.yaml"

# ── 1. Tìm hoặc cài Conda ──────────────────────────────────────────────
find_conda() {
    # Thử các vị trí phổ biến
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "/opt/conda/bin/conda" \
        "/usr/local/conda/bin/conda" \
        "$(which conda 2>/dev/null)"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

CONDA_BIN=$(find_conda) || true

if [ -z "$CONDA_BIN" ]; then
    echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
    MINICONDA_DIR="$HOME/miniconda3"
    MINICONDA_SH="/tmp/miniconda_install.sh"

    # Chọn đúng installer theo hệ điều hành & kiến trúc
    OS_TYPE=$(uname -s)
    ARCH=$(uname -m)
    if [ "$OS_TYPE" = "Linux" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    elif [ "$OS_TYPE" = "Darwin" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${ARCH}.sh"
    else
        echo -e "${RED}ERROR: Unsupported OS: $OS_TYPE${NC}"
        exit 1
    fi

    echo "  Downloading: $MINICONDA_URL"
    curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_SH" || wget -q "$MINICONDA_URL" -O "$MINICONDA_SH"
    bash "$MINICONDA_SH" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_SH"

    CONDA_BIN="$MINICONDA_DIR/bin/conda"
    echo -e "${GREEN}✓ Miniconda installed at: $MINICONDA_DIR${NC}"
fi

echo -e "${GREEN}✓ Conda found: $CONDA_BIN${NC}"
echo "  Version: $($CONDA_BIN --version)"

# Khởi tạo conda trong shell hiện tại
CONDA_BASE=$($CONDA_BIN info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || \
    eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" || true

# ── 2. Tạo hoặc cập nhật môi trường conda ─────────────────────────────
if ! $CONDA_BIN env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
    echo -e "${BLUE}Creating conda environment '${CONDA_ENV_NAME}' from ${CONDA_ENV_FILE}...${NC}"
    echo "  (This may take 5-15 minutes the first time)"
    $CONDA_BIN env create -f "$CONDA_ENV_FILE" --name "$CONDA_ENV_NAME" || {
        echo -e "${YELLOW}⚠ Conda env create failed — trying pip fallback...${NC}"
        pip install -q -r "$PROJECT_DIR/requirements.txt" || true
    }
else
    echo -e "${YELLOW}Conda env '${CONDA_ENV_NAME}' already exists — updating...${NC}"
    $CONDA_BIN env update -f "$CONDA_ENV_FILE" --name "$CONDA_ENV_NAME" --prune || true
fi

# ── 3. Kích hoạt môi trường ────────────────────────────────────────────
conda activate "$CONDA_ENV_NAME" 2>/dev/null || \
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV_NAME" 2>/dev/null || true

# Kiểm tra pytorch_lightning đã có chưa
if python -c "import pytorch_lightning" 2>/dev/null; then
    PL_VER=$(python -c "import pytorch_lightning; print(pytorch_lightning.__version__)")
    echo -e "${GREEN}✓ pytorch_lightning $PL_VER ready${NC}"
else
    echo -e "${YELLOW}pytorch_lightning not found in conda env — installing via pip...${NC}"
    pip install -q pytorch-lightning==1.9.5 huggingface_hub==0.25.2 safetensors gdown || true
fi

echo -e "${GREEN}✓ Environment ready: ${CONDA_ENV_NAME}${NC}"

# ========================
# DOWNLOAD DATA FROM GOOGLE DRIVE
# ========================
if [ "$SKIP_DOWNLOAD" = false ] && [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo -e "${BLUE}Downloading training data from Google Drive...${NC}"
    echo "  Folder ID: $GDRIVE_FOLDER_ID"
    echo "  Destination: $DATA_DIR"

    # Install gdown if not available
    if ! python -c "import gdown" 2>/dev/null; then
        echo -e "${YELLOW}Installing gdown...${NC}"
        pip install -q gdown
    fi

    # Download the entire folder
    python << 'PYTHON_END'
import gdown
import os
import sys

folder_id = os.environ.get('GDRIVE_FOLDER_ID', '1rNGGW8pHQZmCNxL1QyAC_ToswUtpWsjP')
output_dir = os.environ.get('DATA_DIR', './data')

url = f"https://drive.google.com/drive/folders/{folder_id}"
print(f"Downloading from: {url}")
print(f"Saving to: {output_dir}")

try:
    gdown.download_folder(
        url=url,
        output=output_dir,
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    # Count downloaded images
    count = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                count += 1
    print(f"\n✓ Download complete! {count} images found in {output_dir}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("")
    print("Troubleshooting tips:")
    print("  1. Make sure the Google Drive folder is publicly accessible")
    print("  2. Try: pip install -U gdown")
    print("  3. Manual download: " + url)
    sys.exit(1)
PYTHON_END

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Data download failed${NC}"
        echo -e "Please download manually from: $GDRIVE_URL"
        echo -e "and place images in: $DATA_DIR"
        read -p "Continue without downloading? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Data downloaded to: $DATA_DIR${NC}"
    fi
else
    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo -e "${YELLOW}⚠ Skipping data download (--skip-download flag set)${NC}"
    else
        echo -e "${GREEN}✓ Data directory already has content, skipping download${NC}"
    fi
fi

# Set DATA_ROOT to DATA_DIR if DATA_ROOT was not explicitly provided
if [ "$DATA_ROOT" = "." ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    DATA_ROOT="$DATA_DIR"
    echo -e "${BLUE}Using DATA_ROOT: $DATA_ROOT${NC}"
fi

# ========================
# VERIFY DATA DIRECTORY
# ========================
if [ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ WARNING: Data directory is empty: $DATA_ROOT${NC}"
    echo -e "Please provide training images in: $DATA_ROOT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Training cancelled${NC}"
        exit 0
    fi
else
    echo -e "${GREEN}✓ Data directory found: $DATA_ROOT${NC}"
    echo "  Images: $(find "$DATA_ROOT" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)"
fi

# ========================
# DOWNLOAD STABLE DIFFUSION 1.5 MODEL
# ========================
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Downloading Stable Diffusion 1.5 model...${NC}"
    echo "URL: $SD_DOWNLOAD_URL"
    echo "(This may take 5-10 minutes and ~5GB of storage)"

    # Try huggingface_hub first
    if python -c "import huggingface_hub" 2>/dev/null; then
        echo -e "${BLUE}Using huggingface_hub...${NC}"
        python << 'PYTHON_END'
from huggingface_hub import hf_hub_download
import os

model_dir = os.path.expanduser("./models/sd1.5")
os.makedirs(model_dir, exist_ok=True)

try:
    file_path = hf_hub_download(
        repo_id="sd-legacy/stable-diffusion-v1-5",
        filename="v1-5-pruned.ckpt",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"✓ Downloaded to: {file_path}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
PYTHON_END
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Trying wget fallback...${NC}"
            cd "$MODEL_DIR" || exit 1
            wget -c "$SD_DOWNLOAD_URL" -O "$MODEL_NAME" || {
                echo -e "${RED}ERROR: Download failed. Please download manually:${NC}"
                echo "  $SD_DOWNLOAD_URL"
                echo "and save to: $MODEL_PATH"
                exit 1
            }
            cd - > /dev/null
        fi
    else
        echo -e "${BLUE}Using wget...${NC}"
        cd "$MODEL_DIR" || exit 1
        wget -c "$SD_DOWNLOAD_URL" -O "$MODEL_NAME" || {
            echo -e "${RED}ERROR: Download failed${NC}"
            exit 1
        }
        cd - > /dev/null
    fi
else
    echo -e "${GREEN}✓ Model already exists: $MODEL_PATH${NC}"
fi

# ========================
# CHECK GPU
# ========================
echo -e "${BLUE}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi -L 2>/dev/null)
    echo "$GPU_INFO" | head -1
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected${NC}"
fi

# ========================
# SHOW TRAINING CONFIG
# ========================
# echo -e "${BLUE}"
# echo "╔════════════════════════════════════════════════════╗"
# echo "║           TRAINING CONFIGURATION                   ║"
# echo "╚════════════════════════════════════════════════════╝"
# echo -e "${NC}"
# echo "  Run Name:   $RUN_NAME"
# echo "  Data Root:  $DATA_ROOT"
# echo "  Model:      $MODEL_PATH"
# echo "  Config:     $CONFIG"
# echo "  GPU:        $GPU_ID"
# echo "  Output:     $OUTPUT_DIR"
# echo "  Logs:       $LOG_DIR"
# echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled${NC}"
    exit 0
fi

# ========================
# TRAIN
# ========================
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting training...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

python main.py \
  --base "$CONFIG" \
  -t \
  --no-test \
  --actual_resume "$MODEL_PATH" \
  -n "$RUN_NAME" \
  --gpus "$GPU_ID", \
  --data_root "$DATA_ROOT" \
  --logdir "$LOG_DIR" \
  --save_interval 500 \
  --save_top_k 3 \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
else
    echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT_CODE${NC}"
    echo "Check logs: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

# ========================
# FIND LATEST CHECKPOINT
# ========================
echo -e "${BLUE}Finding latest checkpoint...${NC}"

CKPT_DIR="$LOG_DIR/$RUN_NAME/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(find "$CKPT_DIR" -name "*.ckpt" -o -name "*.pt" | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        CKPT_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
        echo -e "${GREEN}✓ Latest checkpoint found: $(basename $LATEST_CKPT) ($CKPT_SIZE)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Checkpoint directory not found${NC}"
    exit 1
fi

# ========================
# EXTRACT EMBEDDING ONLY
# ========================
echo -e "${BLUE}Extracting embeddings from checkpoint...${NC}"

EMBEDDING_DIR="$OUTPUT_DIR/embeddings"
mkdir -p "$EMBEDDING_DIR"

python << 'PYTHON_END'
import torch
from safetensors.torch import save_file
import json
import os

ckpt_path = """$LATEST_CKPT"""
output_dir = """$EMBEDDING_DIR"""

try:
    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    print(f"Total state dict keys: {len(state)}")

    # Extract embedding manager & attention parameters
    embedding_dict = {}
    for key, value in state.items():
        # Keep embedding_manager and attention_transfer params
        if 'embedding_manager' in key or 'attention_transfer' in key:
            clean_key = key.replace('model.', '', 1)
            embedding_dict[clean_key] = value

    if not embedding_dict:
        print("WARNING: No embedding parameters found. Checking all keys...")
        for key in list(state.keys())[:10]:
            print(f"  {key}")

    # Save as safetensors
    embedding_path = os.path.join(output_dir, "embeddings.safetensors")
    save_file(embedding_dict, embedding_path)

    embedding_size_mb = os.path.getsize(embedding_path) / 1024 / 1024
    print(f"✓ Embedding saved: {embedding_path}")
    print(f"  Keys: {len(embedding_dict)}")
    print(f"  Size: {embedding_size_mb:.2f} MB")

    # Save metadata
    metadata = {
        "type": "embedding_only",
        "model": "LSAST (SD1.5 Frozen + Attention Training)",
        "num_embedding_params": len(embedding_dict),
        "embedding_keys": list(embedding_dict.keys()),
        "checkpoint_source": os.path.basename(ckpt_path),
        "extraction_method": "safetensors"
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON_END

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Embedding extraction successful!${NC}"
else
    echo -e "${RED}✗ Embedding extraction failed${NC}"
    exit 1
fi

# ========================
# CLEANUP: Remove full checkpoint
# ========================
echo -e "${YELLOW}Cleaning up full checkpoint...${NC}"
if [ -f "$LATEST_CKPT" ]; then
    ORIGINAL_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
    rm -f "$LATEST_CKPT"
    echo -e "${GREEN}✓ Removed: $ORIGINAL_SIZE (full checkpoint)${NC}"
    echo "  Kept only: embeddings.safetensors (~50MB)"
fi

# ========================
# OPTIONAL: UPLOAD EMBEDDING TO HUGGINGFACE
# ========================
if [ -n "$HF_UPLOAD_REPO" ] && [ -n "$HF_TOKEN" ]; then
    echo -e "${BLUE}Uploading embedding to HuggingFace...${NC}"

    export HF_TOKEN="$HF_TOKEN"

    # Create repo if doesn't exist
    huggingface-cli repo create "$HF_UPLOAD_REPO" --type model 2>/dev/null || true

    EMBEDDING_FILE="$EMBEDDING_DIR/embeddings.safetensors"
    if [ -f "$EMBEDDING_FILE" ]; then
        if command -v hf &> /dev/null; then
            echo -e "${BLUE}Using hf CLI (faster)${NC}"
            hf upload "$HF_UPLOAD_REPO" \
              "$EMBEDDING_FILE" \
              "$EMBEDDING_DIR/metadata.json" \
              --repo-type model \
              --private
        else
            echo -e "${YELLOW}Using huggingface-cli (fallback)${NC}"
            huggingface-cli upload "$HF_UPLOAD_REPO" \
              "$EMBEDDING_FILE" \
              "$EMBEDDING_DIR/metadata.json" \
              --repo-type model
        fi
        echo -e "${GREEN}✓ Upload complete!${NC}"
    fi
fi

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════╗"
echo "║              TRAINING COMPLETE                      ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "Logs saved to:  $LOG_FILE"
echo "Output saved to: $OUTPUT_DIR"
echo ""
