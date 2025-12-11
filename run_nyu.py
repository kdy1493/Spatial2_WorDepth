#!/usr/bin/env python3
"""
NYU Depth V2 Training Script
Replacement for run_nyu.sh
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Configuration
    model_name = "nyu_train"
    config_file = "configs/nyu_train.yaml"
    gpu_id = "0"  # CUDA_VISIBLE_DEVICES
    
    # Create model directory
    model_dir = Path("./models") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file path
    log_file = model_dir / "result.log"
    
    # Set environment variable for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Construct command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "src/train.py",
        config_file
    ]
    
    print(f"Starting NYU Depth V2 training with model: {model_name}")
    print(f"Using GPU: {gpu_id}")
    print(f"Config file: {config_file}")
    print(f"Log file: {log_file}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run training with output redirection
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    print("-" * 80)
    print(f"Training completed with exit code: {process.returncode}")
    print(f"Log saved to: {log_file}")
    
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
