# Environment Setup & Platform Guide

## 1. Hybrid Environment Architecture
* **Local Machine:** Windows 11 with Conda (used for data preparation, reward function authoring, in-memory sandboxing, and local `pytest` validation).
* **Remote GPU Machine:** Google Colab Free Tier (Ubuntu Linux, NVIDIA T4 15GB VRAM, used for baseline generation, 0.5B dry-run, and 1.5B GRPO training).

---

## 2. Local Setup (Windows with Conda)

### Step 1: Create and Activate Conda Environment
Open Anaconda Prompt / PowerShell and run:

```powershell
# Create dedicated Python 3.10 environment
conda create -n grpo-sql python=3.10 -y

# Activate the environment
conda activate grpo-sql
```

### Step 2: Install Local Dependencies
Run the following in the project root directory:

```powershell
pip install -r requirements.txt
```

### Step 3: Run Verification Unit Tests
Verify that the local SQLite sandbox and reward calculations pass 100% of unit tests:

```powershell
pytest tests/ -v
```

---

## 3. Remote GPU Setup (Google Colab T4)

### Step 1: Open Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime $\to$ Change runtime type**.
3. Select **T4 GPU** (Hardware accelerator: GPU, GPU type: T4).

### Step 2: Environment Initialization Cell
Run this cell at the start of your Colab session (`notebooks/colab_train.ipynb`):

```python
# 1. Verify GPU availability
!nvidia-smi

# 2. Clone the repository
!git clone https://github.com/<YOUR_USERNAME>/grpo-sql-post-training.git
%cd grpo-sql-post-training

# 3. Install required packages
!pip install --quiet \
    "transformers>=4.40.0" \
    "trl>=0.14.0" \
    "peft>=0.10.0" \
    "accelerate>=0.28.0" \
    "datasets>=2.18.0" \
    "wandb>=0.16.0" \
    "pytest>=8.0.0"

# 4. Run local test suite inside Colab to verify environment parity
!pytest tests/ -v
```

### Step 3: Mount Google Drive for Checkpoint Persistence
To prevent losing trained weights if Colab disconnects, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Symlink checkpoints directory to Google Drive
!mkdir -p /content/drive/MyDrive/grpo_checkpoints
!ln -s /content/drive/MyDrive/grpo_checkpoints ./checkpoints
```

---

## 4. Known Platform Quirks & How We Avoid Them

### 1. Windows Native C++ Compilation Issues
* **Problem:** Compiling modern LLM libraries like `flash-attn`, `deepspeed`, or native `bitsandbytes` on Windows requires Visual Studio C++ build tools and custom CUDA toolchains, frequently leading to fatal build errors.
* **Our Solution:** We run **zero training code on raw Windows**. All local code uses standard Python library modules (`sqlite3`, `threading`, `json`, `re`) and CPU PyTorch for running unit tests. All GPU compilation happens on standard Ubuntu Linux in Google Colab.

### 2. Colab Idle Disconnects & Preemption
* **Problem:** Free Colab sessions can be preempted or disconnect if left unattended without browser activity.
* **Our Solution:**
  1. Training runs are short (~65 to 75 minutes for 200 steps).
  2. Model checkpoints are saved every 50 steps directly to mounted Google Drive.
  3. All training curves and completion logs are streamed to Weights & Biases in real time, so no metrics are lost if a runtime resets.

### 3. VRAM Headroom on T4 (15 GB)
* **Problem:** Loading a 1.5B model in 32-bit float with full optimizer states causes instant CUDA Out-Of-Memory (OOM) on a 15GB GPU.
* **Our Solution:**
  * We use **16-bit LoRA** (`torch.bfloat16` or `float16`).
  * Base model (3.0 GB) + Reference model (3.0 GB) + LoRA adapters & optimizer (< 0.5 GB) + $G=4$ rollouts (< 2.0 GB) = **~8.5 GB VRAM peak**.
  * This leaves ~6.5 GB of safe headroom on the T4 GPU.
