
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch
BATCH_SIZE = 1

# Prompt
MAX_PROMPT_LENGTH = 512

# Generation
MAX_NEW_TOKENS = 128

# Precision
USE_BFLOAT16 = True

# Random seed
SEED = 42

# Benchmarking
ENABLE_PROFILER = False
ENABLE_LATENCY = True
ENABLE_THROUGHPUT = True

# Hugging Face model path
MODEL_NAME = "inclusionAI/LLaDA-MoE-7B-A1B-Base"

# Local checkpoint (optional)
LOCAL_CHECKPOINT = None