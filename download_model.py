#!/usr/bin/env python
# coding: utf-8
"""
Download Pythia-2.8B model from Hugging Face.

This script downloads the EleutherAI/pythia-2.8b model to a local directory.
The downloaded files will match the structure in pythia-2.8b-local/.
"""

import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download Pythia-2.8B model")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="pythia-2.8b-local",
        help="Directory to save the model (default: pythia-2.8b-local)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch (default: main)"
    )
    args = parser.parse_args()
    
    model_id = "EleutherAI/pythia-2.8b"
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    
    print(f"Downloading {model_id} to {output_dir}...")
    
    # Download model files (safetensors format)
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        # Only download necessary files
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ],
        ignore_patterns=[
            "*.bin",
            "*.h5",
            "*.msgpack",
            "*.ot",
        ],
    )
    
    print(f"\nDownload complete!")
    print(f"Model saved to: {output_dir}")
    print("\nDownloaded files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024 / 1024:.2f} GB)")
        else:
            print(f"  {f} ({size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
