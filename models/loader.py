import os
import glob
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

def get_hf_weights_path(repo_id="inclusionAI/LLaDA-MoE-7B-A1B-Instruct", local_dir="weights"):
    """Downloads weights if they don't exist and returns the path."""
    if not os.path.exists(local_dir) or len(glob.glob(os.path.join(local_dir, "*.safetensors"))) == 0:
        print(f"Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model.ot", "coreml*", "onnx*"]
        )
    return local_dir

def load_hf_weights_into_custom_model(model, weights_dir="weights"):
    """
    Loads HuggingFace safetensors into the custom LLaDAMoE model.
    The custom model parameter names perfectly match the HF ones once the 'model.' prefix is removed.
    """
    print(f"Loading weights from {weights_dir}...")
    safetensor_files = glob.glob(os.path.join(weights_dir, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {weights_dir}")
        
    safetensor_files.sort()
    
    # Load model state dict
    model_state_dict = model.state_dict()
    mapped_keys = 0
    total_keys = len(model_state_dict)
    
    for sf_file in safetensor_files:
        print(f"Loading {os.path.basename(sf_file)}...")
        hf_state_dict = load_file(sf_file)
        
        for k, v in hf_state_dict.items():
            # Map HF keys to custom model keys
            mapped_key = k
            if mapped_key.startswith("model."):
                mapped_key = mapped_key[len("model."):]
                
            if mapped_key in model_state_dict:
                # Load weight directly into the parameter
                with torch.no_grad():
                    # Move to the correct device/dtype if needed, but doing it in memory first
                    model_state_dict[mapped_key].copy_(v)
                mapped_keys += 1
            else:
                print(f"  Warning: Unknown key {k} -> {mapped_key} in HF weights")
                
    print(f"Loaded {mapped_keys}/{total_keys} keys successfully.")
    
    # Check for missing keys
    if mapped_keys < total_keys:
        missing = set(model_state_dict.keys()) - set([
            k[len("model."):] if k.startswith("model.") else k 
            for sf_file in safetensor_files 
            for k in load_file(sf_file).keys()
        ])
        print(f"Missing keys: {missing}")
        
    return model
