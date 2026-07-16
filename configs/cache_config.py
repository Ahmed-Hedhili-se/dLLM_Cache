"""
dLLM-Cache configuration.
"""

# Prompt refresh interval
KP = 50

# Response refresh interval
KR = 7

# Number of denoising steps
NUM_DIFFUSION_STEPS = 128

# Active token ratio (ρ)
UPDATE_RATIO = 0.25

# Cache behaviour
ENABLE_EMBEDDING_CACHE = True
ENABLE_ATTENTION_CACHE = True
ENABLE_MOE_CACHE = True

# Similarity metric
SIMILARITY = "cosine"

# Always refresh first step
FULL_REFRESH_AT_START = True