#!/bin/sh

set -e
set -o xtrace

# first enable the devtoolset-11 environment
# note that deepspeed cpu_adam will be complied at run time, gcc 8 will cause error: RuntimeError: Error building extension 'cpu_adam' in deepspeed 
# source /opt/rh/devtoolset-11/enable can mess up environment variables for CUDA, while error when source scl_source enable devtoolset-8
# Save original CUDA environment variables
# export CUDA_HOME=/usr/local/cuda-11.8
# ORIGINAL_CUDA_HOME="$CUDA_HOME"
# ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
# source /opt/rh/devtoolset-11/enable
# # Restore CUDA environment variables
# export CUDA_HOME="$ORIGINAL_CUDA_HOME"
# export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"

export PATH=$HOME/.local/bin:$PATH
# source scl_source enable devtoolset-11

# pod is not enabled with Interactive Dev, do following within the pod:
# nohup /home/jobuser/.local/lib/code-server-4.16.1-linux-amd64/bin/code-server --bind-addr 0.0.0.0:8080 --auth none &

# install the needed python lib
pip install vllm==0.6.1.post2+cu118  # vllm is installed separately to avoid conflicts with transformers
pip install -r /home/jobuser/controlllm/requirements.txt

# install bitsandbytes
# cd /home/jobuser/resources/bitsandbytes-0.42.0
# CUDA_HOME=/usr/local/cuda-11.8 CUDA_VERSION=118 make cuda11x
# python /home/jobuser/resources/bitsandbytes-0.42.0/setup.py install --user
# cd /home/jobuser

# install lm-evaluation-harness
pip3 install -e /home/jobuser/resources/lm-evaluation-harness

# install llm-alignment
pip3 install -e /home/jobuser/resources/llm-alignment-workspace

# make lm-evaluation-harness work
pip uninstall -y antlr4-python3-runtime
pip install antlr4-python3-runtime==4.11.1  # to make math hard eval work
# for vllm to work
pip uninstall -y prometheus_client
pip install prometheus_client==0.20.0  # to make vllm work
pip uninstall -y jsonschema
pip install jsonschema==4.21.1  # to make vllm work
pip uninstall -y pydantic
pip install pydantic==2.9.2  # to make vllm work
pip uninstall -y pydantic_core
pip install pydantic_core==2.23.4
pip uninstall -y annotated-types
pip install annotated-types==0.7.0
pip uninstall -y triton
pip install triton==3.0.0  # to make vllm work
pip uninstall -y torchvision
pip install torchvision==0.19.0+cu118  # to make vllm work
