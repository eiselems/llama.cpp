#!/bin/bash
# Memory probe for bench.py: prints the GPU memory in use (whole GPU, desktop included) in bytes.
echo $(( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) * 1024 * 1024 ))
