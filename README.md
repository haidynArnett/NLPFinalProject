# CS 4650 Group 15 Final Project

## Multi-Agent Knowledge Transfer Evaluation

This project evaluates information degradation in sequential multi-agent LLM systems through novel metrics that measure semantic drift during agent handoffs.

## Setup Instructions

### 1. Make Scripts Executable
```bash
chmod +x scripts/check_ollama.sh
```

### 2. Create and Activate Conda Environment
```bash
conda env create -f requirements.yml
conda activate nlp-final
```

### 3. Install Package in Development Mode
```bash
pip install -e .
```

### 4. Select Kernel in Jupyter Notebook
After opening `main.ipynb`, select the kernel:

1. Click on "Kernel" in the menu bar
2. Select "Change kernel"
3. Choose "nlp-final" from the list of available kernels

Alternatively, you can run this in a notebook cell:
```python
# To ensure you're using the correct kernel
import sys
print(f"Python executable: {sys.executable}")
```

### 5. Verify Ollama Setup
```bash
./scripts/check_ollama.sh
```

## Project Structure

```
NLPFinalProject/
├── main.ipynb              # Main analysis notebook
├── requirements.yml        # Conda environment specification
├── src/                    # Source code
│   ├── experiments/        # Experiment implementations
│   ├── metrics/            # Evaluation metrics
│   ├── visualization/      # Plotting utilities
│   └── io/                 # Input/Output utilities
└── scripts/                # Utility scripts
```

## Models Tested

- Qwen3:235b
- Qwen3:8b
- Llama3.1:70b
- Llama3.1:7b
- Gemma3:4b
- GPT-OSS:20b